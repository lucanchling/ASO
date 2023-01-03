
import os 
import glob 
import vtk
import numpy as np
import json
<<<<<<< HEAD
=======
from vtk.util.numpy_support import vtk_to_numpy
from torch import tensor
import torch
from random import randint
from math import pi
>>>>>>> 355a436 (to merge luc branch)




def ReadSurf(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    surf = reader.GetOutput()

    return surf

def LoadJsonLandmarks(ldmk_path,full_landmark=True,list_landmark=[]):
    """
    Load landmarks from json file
    
    Parameters
    ----------
    img : sitk.Image
        Image to which the landmarks belong
 
    Returns
    -------
    dict
        Dictionary of landmarks
    
    Raises
    ------
    ValueError
        If the json file is not valid
    """

    with open(ldmk_path) as f:
        data = json.load(f)
    
    markups = data["markups"][0]["controlPoints"]
    
    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord
    
    if not full_landmark:
        out={}
        for lm in list_landmark:
            out[lm] = landmarks[lm]
        landmarks = out
    return landmarks






<<<<<<< HEAD

def WriteSurf(surf, output_folder,name,inname):
    dir, name = os.path.split(name)
=======
def WriteSurf(surf, output_folder,name,inname):
>>>>>>> 355a436 (to merge luc branch)
    name, extension = os.path.splitext(name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    writer = vtk.vtkPolyDataWriter()
<<<<<<< HEAD
=======
    # print(output_folder,os.path.join(output_folder,name))
>>>>>>> 355a436 (to merge luc branch)
    writer.SetFileName(os.path.join(output_folder,f"{name}{inname}{extension}"))
    writer.SetInputData(surf)
    writer.Update()


def ReadSurf(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    surf = reader.GetOutput()

    return surf





def UpperOrLower(path_filename):
    """tell if the file is for upper jaw of lower

    Args:
        path_filename (str): exemple /home/..../landmark_upper.json

    Returns:
        str: Upper or Lower, for the following exemple if Upper
    """
    out = 'Lower'
    st = '_U_'
    st2= 'upper'
    filename = os.path.basename(path_filename)
    if st in filename or st2 in filename.lower():
        out ='Upper'
    return out




def search(path,*args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

<<<<<<< HEAD
    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments=[]
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}
=======
def manageICP(input,target,list_teeth):
    source = MidTeeth(input,list_teeth)
    target = MidTeeth(target, list_teeth)

    source = SortDict(source)
    target = SortDict(target)

    script_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(script_dir,'cache')):
        os.mkdir(os.path.join(script_dir,'cache'))
    np.save(os.path.join(script_dir ,'cache','source.npy'), source)
    np.save(os.path.join(script_dir ,'cache','target.npy'), source)

    source_transformed, TransformMatrix, TransformList = InitICP(source,target, BestLMList=FindOptimalLandmarks(source,target))
    TransformMatrixBis = first_ICP(source_transformed,target) 


    TransformMatrixFinal = TransformMatrixBis @ TransformMatrix



    return TransformMatrixFinal
>>>>>>> 355a436 (to merge luc branch)




def PatientNumber(filename):
    number = ['1','2','3','4','5','6','7','8','9','0']
    for i in range(len(filename)):
        if filename[i] in number:
            for y in range(i,len(filename)):
                if not filename[y] in number:
                    return int(filename[i:y])





def WriteJsonLandmarks(landmarks,output_file,input_file_json,add_innamefile,output_folder):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    # # Load the input image
    # spacing, origin = LoadImage(input_file)
    dirname , name  = os.path.split(output_file)
    name, extension = os.path.splitext(name)
    output_file = os.path.join(output_folder,name+add_innamefile+extension)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    

    with open(input_file_json, 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']]
        # pos = (pos + abs(inorigin)) * inspacing
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    with open(output_file, 'w') as outfile:

        json.dump(tempData, outfile, indent=4)




def listlandmark2diclandmark(list_landmark):
    upper =[]
    lower=[]
    list_landmark=list_landmark.split(',')
    for landmark in list_landmark:
        if 'U' == landmark[0]:
            upper.append(landmark)
        else :
            lower.append(landmark)

    out ={'Upper':upper,'Lower':lower}

    return out


<<<<<<< HEAD

def WritefileError(file,folder_error,message):
    if not os.path.exists(folder_error):
        os.mkdir(folder_error)
    name = os.path.basename(file)
    name , _ = os.path.splitext(name)
    with open(os.path.join(folder_error,f'{name}Error.txt'),'w') as f:
        f.write(message)
=======
def MidTeeth(surf,list_teeth):
    assert isinstance(surf,vtk.vtkPolyData)

    region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(GetLabelSurface(surf)))),dtype=torch.int64)
    # print('unique',torch.unique(region_id))
    dic = {}

    for tooth in list_teeth:
        crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        verts_crown = torch.tensor(verts[crown_ids])

        verts_crown = torch.mean(verts_crown,0)
        dic[str(tooth)] = verts_crown.cpu().numpy().astype(np.float64)
    # print('dic',dic)
    
    return dic





def VTKICP(source, target):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    #icp.DebugOn()
    icp.SetMaximumNumberOfIterations(20)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(source)
    icpTransformFilter.SetTransform(icp.GetInverse())
    icpTransformFilter.Update()

    transformedSource = icpTransformFilter.GetOutput()


    return transformedSource




# def RandomRotation(surf):
#     rotationAngle = np.random.random()*360.0
#     rotationVector = np.random.random(3)*2.0 - 1.0
#     rotationVector = rotationVector/np.linalg.norm(rotationVector)
#     transform = vtk.vtkTransform()
#     transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
#     transformFilter = vtk.vtkTransformPolyDataFilter()
#     transformFilter.SetTransform(transform)
#     transformFilter.SetInputData(surf)
#     transformFilter.Update()
#     surf = transformFilter.GetOutput()


#     return surf, rotationAngle, rotationVector



def RandomRotation(surf):
    alpha, beta , gamma  = np.random.random()*pi, np.random.random()*pi, np.random.random()*pi
    Rx = np.array([[1,0,0],[0,np.cos(alpha),np.sin(alpha)],[0,-np.sin(alpha),np.cos(alpha)]])
    Ry = np.array([[np.cos(beta),0,-np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]])
    Rz = np.array([[np.cos(gamma),np.sin(gamma),0],[-np.sin(gamma),np.cos(gamma),0],[0,0,1]])

    matrix_rotation = np.matmul(Rx,Ry)
    matrix_rotation = np.matmul(matrix_rotation,Rz)

    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())
    points = np.matmul(matrix_rotation,points.T).T


    vpoints= vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])
    surf.SetPoints(vpoints)

    return surf, matrix_rotation




def Center(surf):
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())

    points = points - np.mean(points,0)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])


    surf.SetPoints(vpoints)

    return surf


def MeanScale(surf =None ,verts = None):
    if surf : 
        verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()))

    min_coord = torch.min(verts,0)[0]
    max_coord= torch.max(verts,0)[0]
    mean = (max_coord + min_coord)/2.0
    mean= mean.numpy()
    scale = np.linalg.norm(max_coord.numpy() - mean)

    return mean, scale, surf



def PatientNumber(filename):
    number = ['1','2','3','4','5','6','7','8','9','0']
    for i in range(len(filename)):
        if filename[i] in number:
            for y in range(i,len(filename)):
                if not filename[y] in number:
                    return int(filename[i:y])




def GetLabelSurface(surf,Preference='PredictedID'):
    out = None
    string_data = str(surf.GetPointData()).split('\n')
    for i, data in enumerate(string_data) :
        if 'Number Of Arrays:' in data :
            number = [int(i) for i in data.split() if i.isdigit()][0]
            index = i+1
            continue
    list_label = []
    for i in range(index,index+number):
        list_label.append(string_data[i].split('=')[-1])

    if len(list_label)!=0 :
            for label in list_label:
                out = Preference
                if Preference == label:
                    out = Preference
                    continue
                    

    return out
        


def GetMatrixTransform(source,target):
    source = SurfToPoint(source)
    target = SurfToPoint(target)
    RotationTransformMatrix = np.eye(4)
    v1 = abs(source[-1] - source[0])
    v2 = abs(target[1-1] - target[0])
    angle,axis = AngleAndAxisVectors(v2, v1)
    R = RotationMatrix(axis,angle)

    T = target[0] - source[0]

    RotationTransformMatrix[:3, :3] = R
    RotationTransformMatrix[:3, 3] = T


    return RotationTransformMatrix


def SurfToPoint(surf):
    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())


    return points
>>>>>>> 355a436 (to merge luc branch)
