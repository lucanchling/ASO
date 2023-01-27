import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from utils.icp import vtkMeanTeeth
from utils.transformation import RotationMatrix, TransformSurf



cross = lambda a,b: np.cross(a,b)

def make_vector(points2,point1):
    perpen = points2[1]-points2[0]
    perpen= perpen/np.linalg.norm(perpen)

    vector1 = points2[0] - point1
    vector1 = vector1/np.linalg.norm(vector1)

    vector2 = points2[1] - point1
    vector2 = vector2/np.linalg.norm(vector2)

    normal = cross(vector1,vector2)
    normal = normal/np.linalg.norm(normal)

    direction = cross(normal,perpen)
    direction = direction/np.linalg.norm(direction)
    return normal ,direction

def organizeLandmark(landmarks : list):
    assert isinstance(landmarks,list)
    assert len(landmarks) == 3 

    out = {'left':'','middle':'','right':''}

    toothTonumber = {'UR8': '1', 'UR7': '2', 'UR6': '3', 'UR5': '4', 'UR4': '5', 'UR3': '6', 'UR2': '7', 'UR1': '8', 'UL1': '9', 'UL2': '10', 'UL3': '11', 'UL4': '12', 'UL5': '13', 
    'UL6': '14', 'UL7': '15', 'UL8': '16', 'LL8': '17', 'LL7': '18', 'LL6': '19', 'LL5': '20', 'LL4': '21', 'LL3': '22', 'LL2': '23', 'LL1': '24', 'LR1': '25', 'LR2': '26', 'LR3': '27',
     'LR4': '28', 'LR5': '29', 'LR6': '30', 'LR7': '31', 'LR8': '32'}

    dic = {'Upper':{'left':['1','2','3','4','5','6'],'middle':['7','8','9','10'],'right':['11','12','13','14','15','16']},
            'Lower':{'left':['32','31','30','29','28','27'],'middle':['26','25','24','23'],'right':['22','21','20','19','18','17']}}

    if isinstance(landmarks[0],int):
        landmarks = [str(landmark) for landmark in landmarks]

    if not landmarks[0].isdigit():
        landmarks = [ toothTonumber[tooth] for tooth in landmarks]

    if int(landmarks[0]) <= 16 :
        jaw = 'Upper'
    else :
        jaw = 'Lower'
    for side in ['left','middle','right']:
        for landmark in landmarks:
            if landmark in  dic[jaw][side] :
                out[side] = landmark


    return out['left'], out['middle'], out['right']




def PrePreAso(source,target,landmarks):
    left , middle, right = organizeLandmark(landmarks)

    meanTeeth = vtkMeanTeeth([int(left),int(middle),int(right)],property='PredictedID')

    mean_source = meanTeeth(source)
    mean_target = meanTeeth(target)

    normal_source, direction_source = make_vector([mean_source[right],mean_source[left]],mean_source[middle])
    normal_target , direction_target = make_vector([mean_target[right],mean_target[left]],mean_target[middle])



    dt = np.dot(normal_source,normal_target)
    if dt > 1.0 :
        dt = 1.0

    angle_normal = np.arccos(dt)


    normal_normal = cross(normal_source,normal_target)



    matrix_normal = RotationMatrix(normal_normal,angle_normal)
    
    
    
    direction_source = np.matmul(matrix_normal,direction_source.T).T
    direction_source = direction_source / np.linalg.norm(direction_source)

    
    direction_normal = cross(direction_source,direction_target)

    dt = np.dot(direction_source,direction_target)
    if dt > 1.0:
        dt = 1.0

    angle_direction = np.arccos(dt)
    matrix_direction = RotationMatrix(direction_normal ,angle_direction)




    # median_source = np.median(vtk_to_numpy(source.GetPoints().GetData()),axis=0)
    # median_target = np.median(vtk_to_numpy(target.GetPoints().GetData()),axis=0)

    median_source = np.median(np.array([mean_source[right],mean_source[left],mean_source[middle]]),axis=0)
    median_target = np.median(np.array([mean_target[right],mean_target[left],mean_target[middle]]),axis=0)

    mean = median_target-median_source


    mean = np.expand_dims(mean,axis=0)
    matrix_translation = np.concatenate((np.identity(3),mean.T),axis=1)
    matrix_translation = np.concatenate((matrix_translation,np.array([[0,0,0,1]])),axis=0)

    matrix = np.matmul(matrix_direction, matrix_normal)

    matrix = np.concatenate((matrix,np.array([[0,0,0]]).T),axis=1)
    matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0)


    matrix = np.matmul(matrix,matrix_translation)



    



    output = vtk.vtkPolyData()
    output.DeepCopy(source)

    
    output = TransformSurf(output,matrix)


    return output , matrix
