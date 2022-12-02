import vtk 
from utils.utils import ReadSurf, PatientNumber
from vtk.util.numpy_support import vtk_to_numpy

path = '/home/luciacev/Desktop/Data/ALI_IOS/test_prediction_Seg/challenge_teeth_all_vtk/test/QTDZUUZV_upper.vtk'

input = ReadSurf(path)

a = input.GetPointData().__str__().split('\n')
for i, b in enumerate(a) :
    if 'Number Of Arrays:' in b :
        number = [int(i) for i in b.split() if i.isdigit()][0]
        index = i+1
        continue
l = []
for i in range(index,index+number):
    print(a[i])
    l.append(a[i].split('=')[-1])


print('1',a)
print('2',l)