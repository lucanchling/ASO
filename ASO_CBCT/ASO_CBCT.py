#!/usr/bin/env python-real

import sys
import os
import glob
import time
import argparse
import numpy as np
from utils.utils import ( UpperOrLower, search,
 manageICP, ReadSurf, TransformVTKSurf, WriteSurf,VTKICP)




def main(args):
    lower  = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
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


    for file in files:

        jaw = UpperOrLower(file)


        input = ReadSurf(file)
        
        matrix = manageICP(input,gold[jaw],dic_teeth[jaw],args.label_surface[0])
        if True in np.isnan(matrix):
            print('Matrix is not valid',matrix,file)
            continue
            

        print('matrix',matrix)
        
        output1 = TransformVTKSurf(matrix,input)
        output2 = VTKICP(output1,gold[jaw])
        WriteSurf(output2,args.output_folder[0],os.path.basename(file))

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)




  

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()


    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_teeth',nargs=1)
    parser.add_argument('label_surface',nargs=1)


    # parser.add_argument('--input',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/input_test')
    # parser.add_argument('--gold',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/gold')
    # parser.add_argument('--output',default='/home/luciacev/Desktop/Data/ASO_IOS/new_ASO/output')
    # parser.add_argument('--list_teeth',default='30,29,28,21,20,19') #30,29,28,21,20,19



    args = parser.parse_args()





    main(args)