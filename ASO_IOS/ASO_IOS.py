#!/usr/bin/env python-real

import sys
import os
import glob
import time
import json
import argparse

from slicer.util import pip_install


try :
    import vtk
except ImportError:
    pip_install('vtk')

try :
    import numpy as np
except ImportError:
    pip_install('numpy')
    import numpy as np



try :
    import SimpleITK as sitk
except ImportError:
    pip_install('SimpleITK')
    import SimpleITK as sitk

# import vtk
# import numpy as np
# import SimpleITK as sitk

from vtk.util.numpy_support import vtk_to_numpy



def ICP_Transform(source, target):

    # ============ create source points ==============
    source = ConvertToVTKPoints(source)

    # ============ create target points ==============
    target = ConvertToVTKPoints(target)



    # ============ create ICP transform ==============
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(1000)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()


    # ============ apply ICP transform ==============
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(source)
    transformFilter.SetTransform(icp)
    transformFilter.Update()

    return source,target,icp

def first_ICP(source,target):
    source,target,icp = ICP_Transform(source,target)

    return VTKMatrixToNumpy(icp.GetMatrix())

'''
8888888  .d8888b.  8888888b.      8888888 888b    888 8888888 88888888888 
  888   d88P  Y88b 888   Y88b       888   8888b   888   888       888     
  888   888    888 888    888       888   88888b  888   888       888     
  888   888        888   d88P       888   888Y88b 888   888       888     
  888   888        8888888P"        888   888 Y88b888   888       888     
  888   888    888 888              888   888  Y88888   888       888     
  888   Y88b  d88P 888              888   888   Y8888   888       888     
8888888  "Y8888P"  888            8888888 888    Y888 8888888     888
'''

def InitICP(source,target, BestLMList=None, search=False):
    TransformList = []
    # TransformMatrix = np.eye(4)
    TranslationTransformMatrix = np.eye(4)
    RotationTransformMatrix = np.eye(4)

    labels = list(source.keys())
    if BestLMList is not None:
        firstpick, secondpick, thirdpick = BestLMList[0], BestLMList[1], BestLMList[2]


    # ============ Pick a Random Landmark ==============
    if BestLMList is None:
        firstpick = labels[np.random.randint(0, len(labels))]
        # firstpick = 'LOr'
    # ============ Compute Translation Transform ==============
    T = target[firstpick] - source[firstpick]
    TranslationTransformMatrix[:3, 3] = T
    Translationsitk = sitk.TranslationTransform(3)
    Translationsitk.SetOffset(T.tolist())
    TransformList.append(Translationsitk)
    # ============ Apply Translation Transform ==============
    source = ApplyTranslation(source,T)
    # source = ApplyTransform(source, TranslationTransformMatrix)

    # ============ Pick Another Random Landmark ==============
    if BestLMList is None:
        while True:
            secondpick = labels[np.random.randint(0, len(labels))]
            # secondpick = 'ROr'
            if secondpick != firstpick:
                break

    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[secondpick] - source[firstpick])
    v2 = abs(target[secondpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)



    # ============ Compute Rotation Transform ==============
    R = RotationMatrix(axis,angle)
    # TransformMatrix[:3, :3] = R
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============

    source = ApplyTransform(source, RotationTransformMatrix)

    # ============ Compute Transform Matrix (Rotation + Translation) ==============
    TransformMatrix = RotationTransformMatrix @ TranslationTransformMatrix

    # ============ Pick another Random Landmark ==============
    if BestLMList is None:
        while True:
            thirdpick = labels[np.random.randint(0, len(labels))]
            # thirdpick = 'Ba'
            if thirdpick != firstpick and thirdpick != secondpick:
                break
    
    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[thirdpick] - source[firstpick])
    v2 = abs(target[thirdpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)


    # ============ Compute Rotation Transform ==============
    RotationTransformMatrix = np.eye(4)
    R = RotationMatrix(abs(source[secondpick] - source[firstpick]),angle)
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============

    source = ApplyTransform(source, RotationTransformMatrix)

    # ============ Compute Transform Matrix (Init ICP) ==============
    TransformMatrix = RotationTransformMatrix @ TransformMatrix

    if search:
        return firstpick,secondpick,thirdpick, ComputeMeanDistance(source, target)


    return source, TransformMatrix, TransformList

def ComputeMeanDistance(source, target):
    """
    Computes the mean distance between two point sets.
    
    Parameters
    ----------
    source : dict
        Source landmarks
    target : dict
        Target landmarks
    
    Returns
    -------
    float
        Mean distance
    """
    distance = 0
    for key in source.keys():
        distance += np.linalg.norm(source[key] - target[key])
    distance /= len(source.keys())
    return distance


def FindOptimalLandmarks(source,target):
    '''
    Find the optimal landmarks to use for the Init ICP
    
    Parameters
    ----------
    source : dict
        source landmarks
    target : dict
        target landmarks
    
    Returns
    -------
    list
        list of the optimal landmarks
    '''
    dist, LMlist,ii = [],[],0
    script_dir = os.path.dirname(__file__)
    while len(dist) < 210 and ii < 2500:
        ii+=1

        source = np.load(os.path.join(script_dir ,'cache','source.npy'), allow_pickle=True).item()
        firstpick,secondpick,thirdpick, d = InitICP(source,target,search=True)
        if [firstpick,secondpick,thirdpick] not in LMlist:
            dist.append(d)
            LMlist.append([firstpick,secondpick,thirdpick])
    return LMlist[dist.index(min(dist))]

def SortDict(input_dict):
    """
    Sorts a dictionary by key
    
    Parameters
    ----------
    input_dict : dict
        Dictionary to be sorted
    
    Returns
    -------
    dict
        Sorted dictionary
    """
    return {k: input_dict[k] for k in sorted(input_dict)}

                                                                                  

def ConvertToVTKPoints(dict_landmarks):
    """
    Convert dictionary of landmarks to vtkPoints
    
    Parameters
    ----------
    dict_landmarks : dict
        Dictionary of landmarks with key as landmark name and value as landmark coordinates\
        Example: {'L1': [0, 0, 0], 'L2': [1, 1, 1], 'L3': [2, 2, 2]}

    Returns
    -------
    vtkPoints
        VTK points object
    """
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(len(dict_landmarks.keys()))
    labels.SetName("labels")

    for i,landmark in enumerate(dict_landmarks.keys()):
        sp_id = Points.InsertNextPoint(dict_landmarks[landmark])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(sp_id)
        labels.SetValue(i, landmark)
        
    output = vtk.vtkPolyData()
    output.SetPoints(Points)
    output.SetVerts(Vertices)
    output.GetPointData().AddArray(labels)

    return output




def VTKMatrixToNumpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.
    
    Parameters
    ----------
    matrix : vtkMatrix4x4
        Matrix to be copied
    
    Returns
    -------
    numpy array
        Numpy array with the elements of the vtkMatrix4x4
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m






def ConvertTransformMatrixToSimpleITK(transformMatrix):
    '''
    Convert transform matrix to SimpleITK transform
    
    Parameters
    ----------
    transformMatrix : vtkMatrix4x4
        Transform matrix to be converted.
    
    Returns
    -------
    SimpleITK transform
        SimpleITK transform.
    '''
    
    translation = sitk.TranslationTransform(3)
    translation.SetOffset(transformMatrix[0:3,3].tolist())

    rotation = sitk.Euler3DTransform()
    rotation.SetParameters(transformMatrix[0:3,0:3].flatten().tolist())
    # rotation.SetMatrix(transformMatrix[0:3,0:3].flatten().tolist())

    transform = sitk.CompositeTransform([rotation, translation])

    return transform



def ApplyTranslation(source,transform):
    '''
    Apply translation to source dictionary of landmarks

    Parameters
    ----------
    source : Dictionary
        Dictionary containing the source landmarks.
    transform : numpy array
        Translation to be applied to the source.
    
    Returns
    -------
    Dictionary
        Dictionary containing the translated source landmarks.
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = sourcee[key] + transform
    return sourcee

def ApplyTransform(source,transform):
    '''
    Apply a transform matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    transform : np.array
        Transform matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''

    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = transform @ np.append(sourcee[key],1)
        sourcee[key] = sourcee[key][:3]
    return sourcee

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
    """
    import math
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def AngleAndAxisVectors(v1, v2):
    '''
    Return the angle and the axis of rotation between two vectors
    
    Parameters
    ----------
    v1 : numpy array
        First vector
    v2 : numpy array
        Second vector
    
    Returns
    -------
    angle : float
        Angle between the two vectors
    axis : numpy array
        Axis of rotation between the two vectors
    '''
    # Compute angle between two vectors
    v1_u = v1 / np.amax(v1)
    v2_u = v2 / np.amax(v2)
    angle = np.arccos(np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u)))
    cross = lambda x,y:np.cross(x,y)
    axis = cross(v1_u, v2_u)
    #axis = axis / np.linalg.norm(axis)
    return angle,axis



def ReadSurf(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    surf = reader.GetOutput()

    return surf



def TransformVTKSurf(matrix,surf,deepcopy=False):
    if deepcopy:
        surf_copy = vtk.vtkPolyData()
        surf_copy.DeepCopy(surf)
        surf = surf_copy

    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())
 
    # points = points+matrix[:3,3]
    a = np.ones((points.shape[0],1))

    
    # a  = np.expand_dims(a,1)
    points = np.hstack((points,a))

    matrix = matrix[:3,:]

    points = np.transpose(points)
    points = np.matmul(matrix ,points)
    points = np.transpose(points)

    # points = points+matrix[:3,3]



    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])


    surf.SetPoints(vpoints)

    return surf



def WriteSurf(surf, output,add_innamefile,output_folder):
    dirname , name  = os.path.split(output)
    name, extension = os.path.splitext(name)
    output_file = os.path.join(output_folder,name+add_innamefile+extension)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    writer = vtk.vtkPolyDataWriter()

    writer.SetFileName(output_file)
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
    st = 'upper'
    filename = os.path.basename(path_filename)
    if st in filename.lower():
        out ='Upper'
    return out


def PatientNumber(filename):
    number = ['1','2','3','4','5','6','7','8','9','0']
    for i in range(len(filename)):
        if filename[i] in number:
            for y in range(i,len(filename)):
                if not filename[y] in number:
                    return int(filename[i:y])

def LoadJsonLandmarks(ldmk_path,orientation=True,dic_landmark=[]):
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
    # Upper = ['UR6O','UR1O','UL1O','UL6O']
    # Lower = [ 'LR6O','LR1O','LL1O','LL6O']
    Upper = ['UR6O','UR1O','UL6O']
    Lower = [ 'LR6O','LR1O','LL6O']
    
    dic ={'Upper':Upper,'Lower':Lower}
    jaw = UpperOrLower(os.path.basename(ldmk_path))
    with open(ldmk_path) as f:
        data = json.load(f)
    
    markups = data["markups"][0]["controlPoints"]
    
    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
        #lm_coord = ((lm_ph_coord - origin) / spacing).astype(np.float16)
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord

    if orientation:
        out={}
        for lm in dic_landmark[jaw]:
            out[lm] = landmarks[lm]
        landmarks = out
    return landmarks

def manageICP(input_json,gold_json,diclandmark):
    source = LoadJsonLandmarks(input_json,orientation=True,dic_landmark=diclandmark)
    target = LoadJsonLandmarks(gold_json,orientation=True,dic_landmark=diclandmark)

    source = SortDict(source)
    source_orig = source.copy()
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


def search(path,type):
    out=[]
    a = glob.glob(path+'/*'+type)
    for p in a: 
        if os.path.isfile(p):
            out.append(p)
        else:
            out+= search(p,type)
    return out

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

def main(args):


    dic_landmark=listlandmark2diclandmark(args.list_landmark[0])
    print(dic_landmark)




    dic_gold={}
    gold_json = glob.glob(args.gold_folder[0]+'/*json')

    dic_gold[UpperOrLower(gold_json[0])]= gold_json[0]
    dic_gold[UpperOrLower(gold_json[1])]= gold_json[1]




    json_files = search(args.input[0],'json')
    vtk_files = search(args.input[0],'vtk')
    list_file=[]
    iter = 0 
    for json_file in json_files :
        json_name = os.path.basename(json_file)

        json_jaw = UpperOrLower(json_file)
        json_id = PatientNumber(json_name)
        for vtk_file in vtk_files:
            iter+=1
            vtk_name = os.path.basename(vtk_file)
            vtk_id = PatientNumber(vtk_name)
            vtk_jaw = UpperOrLower(vtk_file)

            if vtk_id==json_id and vtk_jaw == json_jaw:
                list_file.append({'json':json_file,'vtk':vtk_file})
                vtk_files.remove(vtk_file)




                sys.stdout.flush()


    print('start icp')
    iter = 0
    for file in list_file:
        iter+=1
        print(iter,file)
        jaw = UpperOrLower(file['json'])
        try :
            matrix = manageICP(file['json'],dic_gold[jaw],dic_landmark)
        except KeyError:
            print('error, KeyError')
            continue
        landmark = LoadJsonLandmarks(file['json'],orientation=False)
        landmark = ApplyTransform(landmark,matrix)

        surf_input = ReadSurf(file['vtk'])
        surf_output = TransformVTKSurf(matrix,surf_input)

        WriteJsonLandmarks(landmark, file['json'],dic_gold[jaw],args.add_inname[0],args.output_folder[0])
        WriteSurf(surf_output,file['vtk'],args.add_inname[0],args.output_folder[0])

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)





  

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()


    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_landmark',nargs=1)


    args = parser.parse_args()





    main(args)