
import os 
import glob 
import vtk
import numpy as np
import SimpleITK as sitk
import json
from vtk.util.numpy_support import vtk_to_numpy
from torch import tensor
import torch
from random import randint

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

    points = np.matmul(matrix ,points.T).T


    # points = points+matrix[:3,3]



    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])


    surf.SetPoints(vpoints)

    return surf



def WriteSurf(surf, output_folder,name):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    writer = vtk.vtkPolyDataWriter()
    print(output_folder,os.path.join(output_folder,name))
    writer.SetFileName(os.path.join(output_folder,name))
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
    st2= 'Upper'
    filename = os.path.basename(path_filename)
    if st in filename or st2 in filename:
        out ='Upper'
    return out





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




def search(path,extension):
    out =[]
    files = glob.glob(os.path.join(path,extension))
    folders = os.listdir(path)
    for file in files:
        out.append(file)
    for folder  in folders:
        if os.path.isdir(os.path.join(path,folder)):
            out+=search(os.path.join(path,folder),extension)

    return out


def MidTeeth(surf,list_teeth):
    region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
    dic = {}
    middl = False
    for tooth in list_teeth:
        crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        verts_crown = torch.tensor(verts[crown_ids])

        if middl:
            verts_max = torch.max(verts_crown,0).values
            verts_min = torch.min(verts_crown,0).values


            middle = tensor(verts_max + verts_min) / 2
            dic[str(tooth)] = middle.cpu().numpy().astype(np.float64)
        else :
            verts_crown = torch.mean(verts_crown,0)
            dic[str(tooth)] = verts_crown.cpu().numpy().astype(np.float64)
    
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













