#!/usr/bin/env python-real

import sys
import os
import glob
import time
import argparse
import numpy as np
from utils.utils import ( UpperOrLower, search,
 manageICP, ReadSurf, TransformVTKSurf, WriteSurf,VTKICP,RandomRotation,Center,MeanScale,PatientNumber,GetMatrixTransform)
from tqdm import tqdm



def main(args):
    lower  = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    dic_teeth={'Upper':[],'Lower':[]}

    list_teeth = args.list_teeth[0].split(',')

    for tooth in list_teeth:
        if int(tooth) in lower :
            dic_teeth['Lower'].append(int(tooth))
        else :
            dic_teeth['Upper'].append(int(tooth))



    gold_files = glob.glob(args.gold_folder[0]+'/*vtk')

    gold ={}

    gold[UpperOrLower(gold_files[0])]= ReadSurf(gold_files[0])
    gold[UpperOrLower(gold_files[1])]= ReadSurf(gold_files[1])





    files = search(args.input[0],'*.vtk')

    list_files=[]


    files_upper =[]
    files_lower = []

    for file in files :
        jaw  = UpperOrLower(file)
        if jaw == "Upper":
            files_upper.append(file)
        else :
            files_lower.append(file)


    for upper in files_upper:
        upper_name = os.path.basename(upper)
        upper_id = PatientNumber(upper_name)


        for lower in files_lower:

            lower_name = os.path.basename(lower)
            lower_id = PatientNumber(lower_name)

            if lower_id==upper_id:
                list_files.append({"Upper":upper,"Lower":lower})
                files_lower.remove(lower)

                sys.stdout.flush()

            
    jaw = 'Upper'


    for file in tqdm(list_files,total=len(list_files)):
        

        input = ReadSurf(file[jaw])
        
        matrix = manageICP(input,gold[jaw],dic_teeth[jaw])
        stop = 0
        while  True in np.isnan(matrix) and stop<10:
            mean , scale , _ = MeanScale(surf = input)
            print('loop1',file,mean,scale)
            input = Center(input)
            input, _ = RandomRotation(input)
            matrix = manageICP(input,gold[jaw],dic_teeth[jaw])
            mean , scale , _ = MeanScale(surf = input)
            print('loop2',file,mean,scale)
            stop+=1
        if stop>=10:
            print('this file is not orientable',file)
            

            

        # print('matrix',matrix)
        
        
        output_upper = TransformVTKSurf(matrix,input)
        
        # WriteSurf(output1,args.output_folder[0],os.path.basename(file),'mid')
        # output2 = VTKICP(output1,gold[jaw])
        WriteSurf(output_upper,args.output_folder[0],os.path.basename(file[jaw]),args.add_inname[0])

        output_lower = TransformVTKSurf(matrix,ReadSurf(file['Lower']))
        WriteSurf(output_lower,args.output_folder[0],os.path.basename(file['Lower']),args.add_inname[0])

        # print(f"""<filter-progress>{0}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.2)
        # print(f"""<filter-progress>{2}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.2)
        # print(f"""<filter-progress>{0}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.2)




  

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()


    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_teeth',nargs=1)


    # parser.add_argument('--input',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/input_test')
    # parser.add_argument('--gold',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/gold')
    # parser.add_argument('--output',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/output')
    # parser.add_argument('--list_teeth',default='30,29,28,21,20,19') #30,29,28,21,20,19



    args = parser.parse_args()





    main(args)