
import os 
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from utils.utils import LoadJsonLandmarks, ReadSurf
from random import choice


class ICP:
    def __init__(self,list_icp,option=None) -> None:
        if  False in [callable(f) for f in list_icp]:
            raise Exception("objects inside of list_icp are not callable")
        self.list_icp=list_icp
        self.option=option

    def copy(self,source):
        if isinstance(source,(dict,list,np.ndarray)):
            source_copy = source.copy()

        if isinstance(source,vtk.vtkPolyData):
            source_copy = vtk.vtkPolyData()
            source_copy.DeepCopy(source)


        return source_copy


    def pathTo(self,source,target):
        assert os.path.isfile(source) and os.path.isfile(target), 'source and target are not file'
        assert os.path.splitext(source)[-1] == os.path.splitext(target)[-1], 'source and target dont have the same extension'

        if source.endswith('.json'):
            source = LoadJsonLandmarks(source)
            target = LoadJsonLandmarks(target)
        
        elif source.endswith('.vtk'):
            source = ReadSurf(source)
            target = ReadSurf(target)

        return source , target

    def run(self,source,target) :
        assert type(source) == type(target), "source and target dont have the same type"
        assert self.list_icp!=None , "give icp methode"


        if isinstance(source,str):
            source , target = self.pathTo(source,target)
    
        source_int = self.copy(source)
        target_int = self.copy(target)

        if callable(self.option):
            source_int = self.option(source_int)
            target_int = self.option(target_int)
    

        matrix_final = np.identity(4)
        
        source_icp = self.copy(source_int)

        for icp in self.list_icp:
            source_icp , matrix = icp(source_icp,target_int)
            matrix_final = matrix_final @ matrix

    

        dic_out={'source':source,'matrix':matrix_final,'source_Or':ApplyTransform(source,matrix_final),'target':target,'source_int':source_int,'source_icp':source_icp,'target_int':target_int}

        return dic_out



        




class vtkICP:
    def setup(self,source,target):
        if isinstance(source,list):
            source = {str(i): np.array(source[i]) for i in range(len(source))}
            target = {str(i): np.array(target[i]) for i in range(len(target))}

        if isinstance(source,dict):
            source = DictTovtkPoints(source)
            target = DictTovtkPoints(target)

        return source , target

    def __call__(self,source,target) :
        assert type(source)== type(target), "source and target dont have the same type"
        

        source, target = self.setup(source,target)

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(100)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()



        # ============ apply ICP transform ==============
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(icp)
        transformFilter.Update()

        return source,VTKMatrixToNumpy(icp.GetMatrix())
    


class InitIcp:
    def setup(self,source,target):
        if isinstance(source,vtk.vtkPolyData):
            source = vtk_to_numpy(source.GetPoints().GetData()).tolist()
            target = vtk_to_numpy(target.GetPoints().GetData()).tolist()

        if isinstance(source ,(list,np.ndarray)):
            source = {str(i): np.array(source[i],dtype=np.single) for i in range(len(source))}
            target = {str(i): np.array(target[i],dtype=np.single) for i in range(len(target))}

        source , target = SameNumberPoint(source,target)

        return source, target



    def __call__(self,source,target):
        source , target = self.setup(source, target)
        assert len(source)==len(target), "dont have the same of point"
        assert len(source)>=3, 'source and target dont have enough point'


        source = {k: source[k] for k in sorted(source)}
        target = {k: target[k] for k in sorted(target)}

        script_dir = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(script_dir,'cache')):
            os.mkdir(os.path.join(script_dir,'cache'))
        np.save(os.path.join(script_dir ,'cache','source.npy'), source)
        np.save(os.path.join(script_dir ,'cache','target.npy'), source)
        best = self.FindOptimalLandmarks(source,target)


        source_transformed, TransformMatrix,= self.InitICP(source,target, BestLMList=best)




        return source_transformed,TransformMatrix


    def InitICP(self,source,target, BestLMList=None, search=False):
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
        
        # ============ Apply Translation Transform ==============
        source = TranslationDict(source,T)
        # source = TransformDict(source, TranslationTransformMatrix)

        # ============ Pick Another Random Landmark ==============
        if BestLMList is None:
            while True:
                secondpick = labels[np.random.randint(0, len(labels))]
                # secondpick = 'ROr'
                if secondpick != firstpick:
                    break

        # ============ Compute Rotation Angle and Axis ==============
        v1 = np.absolute(source[secondpick] - source[firstpick])
        v2 = np.absolute(target[secondpick] - target[firstpick])
        angle,axis = self.AngleAndAxisVectors(v2, v1)



        # ============ Compute Rotation Transform ==============
        R = self.RotationMatrix(axis,angle)
        # TransformMatrix[:3, :3] = R
        RotationTransformMatrix[:3, :3] = R
       
        # ============ Apply Rotation Transform ==============

        source = TransformDict(source, RotationTransformMatrix)

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
        angle,axis = self.AngleAndAxisVectors(v2, v1)


        # ============ Compute Rotation Transform ==============
        RotationTransformMatrix = np.eye(4)
        R = self.RotationMatrix(abs(source[secondpick] - source[firstpick]),angle)
        RotationTransformMatrix[:3, :3] = R

        # ============ Apply Rotation Transform ==============

        source = TransformDict(source, RotationTransformMatrix)

        # ============ Compute Transform Matrix (Init ICP) ==============
        TransformMatrix = RotationTransformMatrix @ TransformMatrix

        if search:
            return firstpick,secondpick,thirdpick, self.ComputeMeanDistance(source, target)


        return source, TransformMatrix



    def FindOptimalLandmarks(self,source,target):
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

        #remplacer 210 by n*(n-1)*(n-2)   (n)
        dist, LMlist,ii = [],[],0
        script_dir = os.path.dirname(__file__)
        n = len(source)
        while len(dist) < n*(n-1)*(n-2) and ii < 2500:
            ii+=1

            source = np.load(os.path.join(script_dir ,'cache','source.npy'), allow_pickle=True).item()
            firstpick,secondpick,thirdpick, d = self.InitICP(source,target,search=True)
            if [firstpick,secondpick,thirdpick] not in LMlist:
                dist.append(d)
                LMlist.append([firstpick,secondpick,thirdpick])

        return LMlist[dist.index(min(dist))]





    def TranslationDict(self,source,transform):
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









    def ComputeMeanDistance(self,source, target):
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


    def RotationMatrix(self,axis, theta):
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

        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        

    def AngleAndAxisVectors(self,v1, v2):
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




class vtkTeeth:
    def __init__(self,list_teeth,property =None):
        self.property = property
        self.list_teeth = list_teeth

    def CheckLabelSurface(self,surf,property):
        if not self.isLabelSurface(surf,property):
            property = self.GetLabelSurface(surf)
        self.property = property
    
    def ListLabelSurface(self,surf):
        string_data = str(surf.GetPointData()).split('\n')
        for i, data in enumerate(string_data) :
            if 'Number Of Arrays:' in data :
                number = [int(i) for i in data.split() if i.isdigit()][0]
                index = i+1
                continue
        list_label = []
        for i in range(index,index+number):
            list_label.append(string_data[i].split('=')[-1])
        for i in range(len(list_label)):
            list_label[i] = list_label[i].strip()

        return list_label



    def GetLabelSurface(self,surf,Preference='Universal_ID'):
        out = None

        list_label = self.ListLabelSurface(surf)

        if len(list_label)!=0 :
            for label in list_label:
                out = label
                if Preference == label:
                    out = Preference
                    continue
                    
        return out



    def isLabelSurface(self,surf,property):
        out = False
        list_label = self.ListLabelSurface(surf)
        if property in list_label:
            out = True
        return out




class vtkIterTeeth(vtkTeeth):
    def __init__(self, list_teeth, surf, property=None):
        super().__init__(list_teeth, property)
        self.CheckLabelSurface(surf,property)

        self.region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        self.verts = vtk_to_numpy(surf.GetPoints().GetData())

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter >= len(self.list_teeth):
            raise StopIteration
        
        verts_crown = np.argwhere(self.region_id==self.list_teeth[self.iter])

        self.iter += 1 
        return np.array(self.verts[verts_crown]) , self.list_teeth[self.iter-1]



class vtkMeanTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self, surf) :
        dic ={}
        for points, tooth in vtkIterTeeth(self.list_teeth,surf,property=self.property):
            dic[str(tooth)]= np.array(np.mean(points,0).squeeze(0))
        return dic


class vtkMiddleTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self,surf):
        dic ={} 
        for points, tooth in vtkIterTeeth(self.list_teeth,surf,property=self.property):
            dic[str(tooth)]= ((np.amax(points,axis=0)+np.amin(points,axis = 0))/2).squeeze(0)
        return dic


class vtkMeshTeeth(vtkTeeth):
    def __init__(self, list_teeth=None, property=None):
        super().__init__(list_teeth, property)
    def __call__(self,surf):
        self.CheckLabelSurface(surf,self.property)
        region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        list_teeth = np.unique(region_id)[1:-1]
        list_points = []
        size = 0

        for points, _ in  vtkIterTeeth(list_teeth,surf,property=self.property):
            list_points.append(points)
            size+= points.shape[0]

    

        Points = vtk.vtkPoints()
        Vertices = vtk.vtkCellArray()
        labels = vtk.vtkStringArray()
        labels.SetNumberOfValues(size)
        labels.SetName("labels")
        index = 0 
        for  points in list_points:
            for i in range(points.shape[0]):
                sp_id = Points.InsertNextPoint(points[i,:].squeeze(0))
                Vertices.InsertNextCell(1)
                Vertices.InsertCellPoint(sp_id)
                labels.SetValue(index, str(index))
                index+=1
            
        output = vtk.vtkPolyData()
        output.SetPoints(Points)
        output.SetVerts(Vertices)
        output.GetPointData().AddArray(labels)

        return output



class SelectKey:
    def __init__(self,list_key) -> None:
        self.list_key = list_key


    def __call__(self,input):
        assert isinstance(input,dict)
        out ={}
        print('selectkey',self.list_key,input)
        for key in self.list_key:
            out[key]=input[key]
        return out






def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())


    points = TransformList(points, matrix)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])


    surf.SetPoints(vpoints)

    return surf




def TransformList(input,matrix):
    type = np.array
    if isinstance(input,list):
        input = np.array(input)
        type = list

    a = np.ones((input.shape[0],1))

    input = np.hstack((input,a))
    matrix = matrix[:3,:]
    input = np.matmul(matrix ,input.T).T

    if isinstance(type,list):
        input = input.tolist()

    return input

    

    



        






def TranslationDict(source,transform):
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








                                                                                  

def DictTovtkPoints(dict_landmarks):
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








def TransformDict(source,transform):
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



def ApplyTransform(input,transform):
    if isinstance(input,vtk.vtkPolyData):
        input = TransformSurf(input,transform)

    if isinstance(input,dict):
        input = TransformDict(input,transform)

    if isinstance(input,(list,np.ndarray)):
        input = TransformList(input,transform)

    return input




def vtkSameNumberPoint(source,target):
    source_points = vtk_to_numpy(source.GetPoints().GetData())
    target_points = vtk_to_numpy(target.GetPoints().GetData())

    if source_points.shape[0] > target_points.shape[0]:
     
        save = np.random.choice(np.arange(0,source_points.shape[0]),target_points.shape[0],replace=False)
        source_points = source_points[save]
        vpoints = vtk.vtkPoints()
        vpoints.SetNumberOfPoints(source_points.shape[0])
        for i in range(source_points.shape[0]):
            vpoints.SetPoint(i,source_points[i])


        source.SetPoints(vpoints)

        

    elif source_points.shape[0] < target_points.shape[0]:
     
        save = np.random.choice(np.arange(0,target_points.shape[0]),source_points.shape[0],replace=False)
        target_points = target_points[save]


        vpoints = vtk.vtkPoints()
        vpoints.SetNumberOfPoints(source_points.shape[0])
        for i in range(target_points.shape[0]):
            vpoints.SetPoint(i,target_points[i])


        target.SetPoints(vpoints)
    
    return source, target


def npSameNumberPoint(source,target):

    if source.shape[0] > target.shape[0]:
     
        save = np.random.choice(np.arange(0,source.shape[0]),target.shape[0],replace=False)
        source = source[save]

        

    elif source.shape[0] < target.shape[0]:
     
        save = np.random.choice(np.arange(0,target.shape[0]),source.shape[0],replace=False)
        target_points = target_points[save]

    
    return source, target



def ListSameNumberPoint(source,target):
    source , target = npSameNumberPoint(np.array(source),np.array(target))

    return source.tolist(), target.tolist()


def DictSameNumberPoint(source,target):
    if choice(list(target.keys())).isdigit():
        source, target =ListSameNumberPoint(list(source.values()),list(target.values()))
        source = {str(i): np.array(source[i]) for i in range(len(source))}
        target = {str(i): np.array(target[i]) for i in range(len(target))}
    return source , target

    


def SameNumberPoint(source,target):
    if isinstance(source,vtk.vtkPolyData):
        source , target = npSameNumberPoint(source,target)

    if isinstance(source,list):
        source , target = ListSameNumberPoint(source,target)
    
    if isinstance(source,dict):
        source , target = DictSameNumberPoint(source,target)


    if isinstance(source,np.ndarray):
        source, target = npSameNumberPoint(source,target)

    return source, target






    