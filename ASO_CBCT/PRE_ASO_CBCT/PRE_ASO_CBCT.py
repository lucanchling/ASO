#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk
import sys,os,time
import torch
import numpy as np

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from utils import (ExtractFilesFromFolder, DenseNet, AngleAndAxisVectors, RotationMatrix)

def ResampleImage(image, transform):
    '''
    Resample image using SimpleITK
    
    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.
        
    Returns
    -------
    SimpleITK image
        Resampled image.
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=int)
    ratio = 1
    new_size = orig_size * ratio
    new_size = np.ceil(new_size).astype(int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    # new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center
    resample.SetOutputOrigin(new_origin)

    return resample.Execute(image)
    
def main(args):

    if not os.path.exists(os.path.split(args.log_path[0])[0]):
        os.mkdir(os.path.split(args.log_path[0])[0])

    with open(args.log_path[0],'w') as log_f :
        log_f.truncate(0)

    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: 1 - CosSim(torch.Tensor(x),torch.Tensor(y))
    
    ckpt_path = os.path.join(args.model_folder[0],'LargeFOV_best.ckpt') # /!\ large and small FOV choice to include /!\ 

    model = DenseNet.load_from_checkpoint(checkpoint_path = ckpt_path)
    model.to('cuda')   
    model.eval()
    
    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    
    input_dir, out_dir = args.input[0], args.output_folder[0]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    input_files = ExtractFilesFromFolder(input_dir, scan_extension)

    for i in range(len(input_files)):
        
        input_file = input_files[i]

        img = sitk.ReadImage(input_file)
        scan = torch.Tensor(sitk.GetArrayFromImage(img)).unsqueeze(0)

        # Translation to center volume
        T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(T.tolist())
        
        goal = np.array((0.0,0.0,1.0)) # Direction vector for good orientation

        with torch.no_grad():
            directionVector_pred = model(scan.to('cuda'))

        directionVector_pred = directionVector_pred.cpu().numpy()

        if Loss(directionVector_pred,goal) > 0.1: # When angle is large enough to apply orientation modification

            angle, axis = AngleAndAxisVectors(goal,directionVector_pred[0])
            Rotmatrix = RotationMatrix(axis,angle)

            rotation = sitk.VersorRigid3DTransform()
            Rotmatrix = np.linalg.inv(Rotmatrix)
            rotation.SetMatrix(Rotmatrix.flatten().tolist())
            
            TransformList = [translation,rotation]
            
            # Compute the final transform (inverse all the transforms)
            TransformSITK = sitk.CompositeTransform(3)
            for i in range(len(TransformList)-1,-1,-1):
                TransformSITK.AddTransform(TransformList[i])
            TransformSITK = TransformSITK.GetInverse()
            
            img_out = ResampleImage(img,TransformSITK)
            
        else: # When angle is too little --> only the center translation is applied

            img_trans = ResampleImage(img,translation.GetInverse())
            img_out = img_trans
        
        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,os.path.basename(input_file))
        if not os.path.exists(file_outpath):
            sitk.WriteImage(img_out, file_outpath)

        with open(args.log_path[0],'r+') as log_f:
            log_f.write(str(i))

if __name__ == "__main__":
    
    print("Starting")
    print(sys.argv)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('input',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('model_folder',nargs=1)
    parser.add_argument('log_path',nargs=1)

    args = parser.parse_args()

    main(args)