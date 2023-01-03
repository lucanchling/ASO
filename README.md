# Automated Standardized Orientation

Automatic Standerize Orientation (ASO) is an extension for 3D Slicer to perform automatic orientation either IOS or CBCT.

## ASO Modules
ASO module provide a convenient user interfacer allowing to oriented different type of scans:
- [CBCT](#aso-cbct) scan
- [IOS](#aso-ios) scan

The implementation is based on iterative closest point's algorithm.


## ASO-CBCT





## ASO-IOS


### Module structure

**Input file:**
The input has to be IOS with teeth's segmentation.
The teeth's segmentation can be automatically done using the [SlicerDentalModelSeg](https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg) extension.

The input is folder containg IOS with the following extension:
```
.vtk
```
 
The IOS files need to have in their name the type of jaw (Upper or Lower).


**Reference :**
The user has to indicate the path of the folder containing [Gold Surface](https://github.com/HUTIN1/ASO/releases/tag/v1.0.1). In this folder has to have one vtk file per jaw.
You can use your own Gold surface or download our Gold surface using the `Download Reference Surface` button in the module `Input section`.




**Landmark selection :**
The user has to choose which teeth he wants to use as reference to oriented. 
We suggest using :
 - Upper jaw : 4,5,6,11,12,13
 - Lower jaw : 19, 20, 21, 27,28,29
This selection applies when you use the `Suggest Teeth` button.


**Label surface:**
The user has to indicate array name of labels in the vtk surface. By default the name is PredictedID.

# Acknowledgements
Authors: Nathan Hutin (University of Michigan), Luc Anchling (University of Michigan), Juan Carlos Prieto (UNC), Lucia Cevidanes (UoM)


# License
It is covered by the Apache License, Version 2.0:

http://www.apache.org/licenses/LICENSE-2.0
