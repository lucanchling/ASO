# Automated Standardized Orientation (ASO)

Automated Standerized Orientation (ASO) is an extension for **3D Slicer** to perform automatic orientation either on IOS or CBCT files.

## ASO Modules

ASO module provide a convenient user interface allowing to orient different type of scans:
- **CBCT** scan
- **IOS** scan

> To select the *Input Type* in the Extension just select between CBCT and IOS here:
> <img src="https://user-images.githubusercontent.com/72148963/216382997-4f4bc446-656b-47ac-811f-c0ed48a92159.png" width="600"/>

## How the module works?

### 2 Modes Available (Semi or Fully Automated)
- **Semi-Automated** (to only run the landmark-based registration with landmark and scans as input)
- **Fully-Automated** (to perform Pre Orientation steps, landmark Identification and ASO with only scans as input)

| Mode | Input |
| ----------- | ----------- |
| Semi-Automated | Scans, Landmark files |
| Fully-Automated | Scans, ALI Models, Pre ASO Models (for **CBCT** files), Segmentation Models (for **IOS** files) |

> To select the *Mode* in the Extension just select between Semi and Fully Automated here:
> <img src="https://user-images.githubusercontent.com/72148963/216383955-0628c1b2-8978-4807-be73-d8029149a4a4.png" width="600"/>

> The **Fully-Automated** Mode `Input` section is slightly different:
> <img src="https://user-images.githubusercontent.com/72148963/216405684-d33d5cce-6964-4f58-9b47-0f8064e9ab46.png" width="600"/>


### Input file:

| Input Type  | Input Extension Type |
| ----------- | ----------- |
| **CBCT** | .nii, .nii.gz, .gipl.gz, .nrrd, .nrrd.gz  |
| **IOS** | .vtk |

> To select the *Input Folder* in the Extension just select your folder with Data here:
> <img src="https://user-images.githubusercontent.com/72148963/216385235-d691a8ea-abcc-47ad-85fa-169ff76d11ec.png" width="600"/>

The input has to be IOS with teeth's segmentation.
The teeth's segmentation can be automatically done using the [SlicerDentalModelSeg](https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg) extension. 
The IOS files need to have in their name the type of jaw (Upper or Lower).

**<ins>Test Files Available:**
You can either download them using the link or  by using the `Download Test Files`.
| Module Selected  | Download Link to Test Files | Information |
| ----------- | ----------- | ----------- |
| **Semi-CBCT** | [Test Files](https://github.com/lucanchling/ASO_CBCT/releases/download/TestFiles/Occlusal_Midsagittal_Test.zip) | Scan and Fiducial List for this [Reference](https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip)|
| **Fully-CBCT** | [Test File](https://github.com/lucanchling/ASO_CBCT/releases/download/TestFiles/Test_File.nii.gz) | Only Scan|
| **Semi-IOS** | [Test Files](https://github.com/HUTIN1/ASO/releases/download/v1.0.2/input_test.zip) | Mesh and Fiducial List [Reference](https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip) |
| **Fully-IOS** | [Test Files](https://github.com/HUTIN1/ASO/releases/download/v1.0.2/input_test.zip)| Only Mesh [Reference](https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip) |

### Reference:

The user has to choose a folder containing a **Reference Gold File** with an oriented scan with landmarks. 
You can either use your own files or download ours using the `Download Reference` button in the module `Input section`.
| Input Type  | Reference Gold Files |
| ----------- | ----------- |
| **CBCT** | [CBCT Reference Files](https://github.com/lucanchling/ASO_CBCT/releases/tag/v01_goldmodels)  |
| **IOS** | [IOS Reference Files](https://github.com/HUTIN1/ASO/releases/tag/v1.0.1) |

> To select the *Reference Folder* in the Extension just select your folder with Reference Data here:
> <img src="https://user-images.githubusercontent.com/72148963/216386412-99f0f39c-6a18-427f-9e0a-c8a20b703602.png" width="600"/>

### Landmark selection 

The user has to decide which **landmarks** he will use to run ASO. 

| Input Type  | Landmarks Available |
| ----------- | ----------- |
| **CBCT** |  Cranial Base, Lower Bones, Upper Bones, Lower and Upper Teeth |
| **IOS** |  Upper and Lower Jaw |

> The landmark selection is handled in the `Option` Section:

For IOS:
  
<img src="https://user-images.githubusercontent.com/72148963/216392364-61fcfe6a-60dd-433d-8364-cf7c4e31d631.png" width="800"/>
  
> - if you select only Lower or Upper, the Upper or Lower jaw will be oriented in first and then the other jaw be oriented using the same transformation. (the morphology of the mouth will be kept)
> - if you select both Lower and Upper, the Upper and Lower will be oriented separately (the morphology of the mouth won't be kept).

For CBCT:

<img src="https://user-images.githubusercontent.com/72148963/216392313-cfae2b21-5194-4ce0-ab18-56ad4eda3d2f.png" width="400"/>

### Models Selection

For the **Fully-Automated** Mode, models are required as input, use the `Download Models` Button or follow the following instructions:
  
#### For CBCT ([Details](https://github.com/lucanchling/ASO#aso-cbct)):
A *Pre-Orientation* and *ALI_CBCT* models are needed
  
> To add the *Pre-Orientation* models just download [PreASOModels.zip](https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip), unzip it and select it here:
> <img src="https://user-images.githubusercontent.com/72148963/216419045-8bee40d9-3421-440a-81e3-255547704d47.png" width="600"/>

> To add the *ALI_CBCT* models go to this [link](https://github.com/Maxlo24/ALI_CBCT/releases/tag/v0.1-models), select the desired models, unzip them in a single folder and select it here:
> <img src="https://user-images.githubusercontent.com/72148963/216420408-33e4269f-dd7f-446c-9186-3ca2b94423ab.png" width="600"/>


#### For IOS:

> To add the *Segmentation model* models just download [model_segmentation.zip](https://github.com/HUTIN1/ASO/releases/download/v1.0.0/model_segmentation.zip), unzip it and select it here:
> <img src="https://user-images.githubusercontent.com/72148963/216421704-884cf7b9-6fdd-40d4-93f6-47437590b7a3.png" width="600"/>

### Outputs Options
> You can decide the *Extension* that the output files will have and the folder where they will go in here:
> <img src="https://user-images.githubusercontent.com/72148963/216422137-e2c4d9a5-43be-45dc-a559-30127008593d.png" width="600"/>

### Let's Run it
> Now that everything is in order, just press the `Run` Button in this section:
> <img src="https://user-images.githubusercontent.com/72148963/216423040-2ba01cf7-848d-4c54-8978-582768d25869.png" width="600"/>


## Algorithm
The implementation is based on iterative closest point's algorithm to execute a landmark-based registration. Some preprocessing steps are done to make the orientation works better (and are described respectively in **CBCT** and **IOS** part)

### ASO CBCT
**Fully-Automated mode:** 
1. a deep learning model is used to predict head orientation and correct it.
Models are available for download ([Pre ASO CBCT Models](https://github.com/lucanchling/ASO_CBCT/releases/tag/v01_preASOmodels))

1. a Landmark Identification Algorithm ([ALI CBCT](https://github.com/DCBIA-OrthoLab/ALI_CBCT)) is used to determine user-selected landmarks

1. an ICP transform is used to match both of the reference and the input file

For the **Semi-Automated** mode, only step **3** is used to match input landmarks with reference's ones.

**<ins>Description of the tool:**
<img width="1244" alt="MethodASO" src="https://user-images.githubusercontent.com/72148963/216373581-67a4915d-912b-4103-b38d-075ef434d133.png">

### ASO IOS

  **Semi-Automated mode:**
 - an ICP transfrom is used to macth both of the reference and the input file by using the landmark
  
  **Fully-Automated mode:**
  
 **<ins> Description of the tool:**

<img width="1058" alt="Screenshot 2023-02-02 at 6 48 50 PM" src="https://user-images.githubusercontent.com/72212416/216476826-01ae5185-27f9-4e26-8c3e-494252c0206a.png">


# Acknowledgements
Nathan Hutin (University of Michigan), Luc Anchling (UoM), Felicia Miranda (UoM), Selene Barone (UoM), Marcela Gurgel (UoM), Najla Al Turkestani (UoM), Juan Carlos Prieto (UNC), Lucia Cevidanes (UoM)


# License
It is covered by the Apache License, Version 2.0:

http://www.apache.org/licenses/LICENSE-2.0
