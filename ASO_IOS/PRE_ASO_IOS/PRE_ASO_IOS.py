#!/usr/bin/env python-real    
import glob
import os
import sys
import time
import argparse


fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from utils import ( UpperOrLower, search, ReadSurf, WriteSurf,PatientNumber,ICP, InitIcp, vtkICP,vtkMeanTeeth,TransformSurf,Files_vtk, Jaw ,Upper, Lower)


print('pre aso ios charge')
    
    
def main(args) :
    print('icp meanteeth launch')

    lower  = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    dic_teeth={'Upper':[],'Lower':[]}

    list_teeth = args.list_teeth[0].split(',')
    dic = {'UR8': 1, 'UR7': 2, 'UR6': 3, 'UR5': 4, 'UR4': 5, 'UR3': 6, 'UR2': 7, 'UR1': 8, 'UL1': 9, 'UL2': 10, 'UL3': 11, 
        'UL4': 12, 'UL5': 13, 'UL6': 14, 'UL7': 15, 'UL8': 16, 'LL8': 17, 'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 
        'LL2': 23, 'LL1': 24, 'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31, 'LR8': 32}

    for tooth in list_teeth:
        if dic[tooth] in lower :
            dic_teeth['Lower'].append(dic[tooth])
        else :
            dic_teeth['Upper'].append(dic[tooth])



    gold_files = glob.glob(args.gold_folder[0]+'/*vtk')

    gold ={}

    gold[UpperOrLower(gold_files[0])]= gold_files[0]
    gold[UpperOrLower(gold_files[1])]= gold_files[1]






    list_files=Files_vtk(args.input[0])
    print('list files', list_files)

    if args.jaw[0] == 'Upper':
        jaw = Jaw(Upper())
    elif args.jaw[0] == 'Lower':
        jaw = Jaw(Lower())
    methode = [ InitIcp(),vtkICP()]
    option = vtkMeanTeeth(dic_teeth[jaw()])
    icp = ICP(methode, option=option)


 
    for i , file in enumerate(list_files):


        output_icp = icp.run(file[jaw()],gold[jaw()])
            

    

        WriteSurf(output_icp['source_Or'],args.output_folder[0],os.path.basename(file[jaw()]),args.add_inname[0])

        surf_lower = ReadSurf(file[jaw.inv()])
        output_lower = TransformSurf(surf_lower,output_icp['matrix'])
        WriteSurf(output_lower,args.output_folder[0],os.path.basename(file[jaw.inv()]),args.add_inname[0])

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
    parser.add_argument('jaw',nargs=1)

    args = parser.parse_args()


    main(args)