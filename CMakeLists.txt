cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ASO)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ASO")
set(EXTENSION_CATEGORY "Automated Dental Tools")
set(EXTENSION_CONTRIBUTORS "Nathan Hutin (University of Michigan), Luc Anchling (University of Michigan)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/ASO.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/ASO/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies


#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ASO)
add_subdirectory(ASO_CBCT)
add_subdirectory(ASO_IOS)
add_subdirectory(ASO_IOS/SEMI_ASO_IOS)
add_subdirectory(ASO_IOS/PRE_ASO_IOS)
add_subdirectory(ASO_IOS/utils)

## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
