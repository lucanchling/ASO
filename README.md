# Automatic Standerize Orientation

Automatic Standerize Orientation (ASO) is an extension for 3D Slicer to perform automatic orient IOS and CBCT.

## ASO Modules
ASO module provide a convenient user interfacer allowing to oriented different type of scans:
- [CBCT](#aso-cbct) scan
- [IOS](#aso-ios) scan

The implementation is based on iterative closest point's algorithm.


## ASO-CBCT





## ASO-IOS


### Module structure

**Input file:**
The input has to be IOS with his landmarks.
The landmark identification can be automatically done using the [SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools) extension.

The input is folder containg IOS with the following extension:
'''
.vtk
'''

The folder need also landmarks with shortcut name of landamrk with the following extension:
'''
.json
'''
 
The IOS and Landmark files need to have in their name the patient's number and type of jaw (Upper or Lower).


**Reference :**
The user has to indicate the path of the folder containing [Gold Landmarks](https://github.com/HUTIN1/ASO/releases/tag/v1.0.0). In this folder has to have one json file per jaw.
You can use your own Gold Landmark or download our Gold Landmark using the `Download Reference Landmark` button in the module `Input section`.

(Gold Landmark )



**Landmark selection :**
The user has to choose which landmarks he wants to use as reference to oriented. 
We suggest using :
 - Upper jaw : UR6o, UR1o, UL6o, UL1o
 - Lower jaw : LR6o, LR1o, LL6o, LL1o
This selection applies when you use the `Suggest Landmark` button.

# Acknowledgment
Authors: Nathan Hutin (University of Michigan), Luc Anchling (University of Michigan), Juan Carlos Prieto (UNC), Lucia Cevidanes (UoM)


# License