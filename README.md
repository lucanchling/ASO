# Automated Standardized Orientation (ASO)

Automated Standerized Orientation (ASO) is an extension for 3D Slicer to perform automatic orientation either on IOS or CBCT files.

## ASO Modules

ASO module provide a convenient user interface allowing to orient different type of scans:
- **CBCT** scan
- **IOS** scan


## How the module works?

### 2 Modes Available (Semi or Fully Automated)
- **Semi-Automated** (to only run the landmark-based registration with landmark and scans as input)
- **Fully-Automated** (to perform Pre Orientation steps, landmark Identification and ASO with only scans as input)

| Mode | Input |
| ----------- | ----------- |
| Semi-Automated | Scans, Landmark files |
| Fully-Automated | Scans, ALI Models, Pre ASO Models (for **CBCT** files), Segmentation Models (for **IOS** files) |


## Module Structure


### Input file:

| Input Type  | Input Extension Type |
| ----------- | ----------- |
| **CBCT** | .nii, .nii.gz, .gipl.gz, .nrrd, .nrrd.gz  |
| **IOS** | .vtk |


> The input has to be IOS with teeth's segmentation.
The teeth's segmentation can be automatically done using the [SlicerDentalModelSeg](https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg) extension. 
The IOS files need to have in their name the type of jaw (Upper or Lower).

### Reference:

The user has to choose a folder containing a **Reference Gold File** with an oriented scan with landmarks. 
You can either use your own files or download ours using the `Download Reference` button in the module `Input section`.
| Input Type  | Reference Gold Files |
| ----------- | ----------- |
| **CBCT** | https://github.com/lucanchling/ASO_CBCT/releases/tag/v01_goldmodels  |
| **IOS** | https://github.com/HUTIN1/ASO/releases/tag/v1.0.1 |


### Landmark selection 

The user has to decide which **landmarks** he will use to run ASO. 

| Input Type  | Landmarks Available |
| ----------- | ----------- |
| **CBCT** |  Cranial Base, Lower Bones, Upper Bones, Lower and Upper Teeth |
| **IOS** |  Upper and Lower Jaw |

>For IOS: The user has to indicate array name of labels in the vtk surface. By default the name is PredictedID.

## Algorithm
The implementation is based on iterative closest point's algorithm to execute a landmark-based registration. Some preprocessing steps are done to make the orientation works better (and are described respectively in **CBCT** and **IOS** part)

### ASO CBCT
**Fully-Automated mode:** 
1. a deep learning model is used to predict head orientation and correct it.
Models are available for download ([Pre ASO CBCT Models](https://github.com/lucanchling/ASO_CBCT/releases/tag/v01_preASOmodels))

1. a Landmark Identification Algorithm ([ALI CBCT](https://github.com/DCBIA-OrthoLab/ALI_CBCT)) is used to determine user-selected landmarks

1. an ICP transform is used to match both of the reference and the input file

> For the **Semi-Automated** mode, only step **3** is used to match input landmarks with reference's ones.
### ASO IOS

# Acknowledgements
Nathan Hutin (University of Michigan), Luc Anchling (UoM), Felicia Miranda (UoM), Selene Barone (UoM), Marcela Gurgel (UoM), Najla Al Turkestani (UoM), Juan Carlos Prieto (UNC), Lucia Cevidanes (UoM)


# License
It is covered by the Apache License, Version 2.0:

http://www.apache.org/licenses/LICENSE-2.0
