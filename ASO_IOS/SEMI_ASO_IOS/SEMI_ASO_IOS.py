#!/usr/bin/env python-real    


import glob
import os
import time
import sys
import argparse
fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)
from utils import (vtkICP, InitIcp, SelectKey, ICP, TransformSurf, UpperOrLower, PatientNumber, 
LoadJsonLandmarks, WriteSurf, WriteJsonLandmarks, search,
 listlandmark2diclandmark, ReadSurf, Files_vtk_json, Jaw, Lower , Upper, ApplyTransform)


print('semi aso ios chargee')

def main(args):
    print('icp landmark launch')


    dic_landmark=listlandmark2diclandmark(args.list_landmark[0])


    dic_gold={}
    gold_json = glob.glob(args.gold_folder[0]+'/*json')

    dic_gold[UpperOrLower(gold_json[0])]= gold_json[0]
    dic_gold[UpperOrLower(gold_json[1])]= gold_json[1]






    list_file=Files_vtk_json(args.input[0])





    methode = [InitIcp(),vtkICP()]
    option_upper = SelectKey(dic_landmark['Upper'])
    option_lower = SelectKey(dic_landmark['Lower'])
    print('dic landmark', dic_landmark)
    icp_upper = ICP(methode,option=option_upper)
    icp_lower = ICP(methode, option=option_lower)
    icp = {'Lower':icp_lower,'Upper':icp_upper}
    print('--------------'*10)

    print(list_file)


    if args.jaw[0] == 'Upper':
        jaw = Jaw(Upper())
    elif args.jaw[0] == 'Lower':
        jaw = Jaw(Lower())
    
    for file in list_file:
        
        print(iter,file)

        try :
            output_icp = icp[jaw()].run(file[jaw()]['json'],dic_gold[jaw()])
        except KeyError:
            print('error  KeyError')
            continue

        surf_input = ReadSurf(file[jaw()]['vtk'])
        surf_output = TransformSurf(surf_input,output_icp['matrix'])

        WriteJsonLandmarks(output_icp['source_Or'], file[jaw()]['json'],file[jaw()]['json'],args.add_inname[0],args.output_folder[0])
        WriteSurf(surf_output,args.output_folder[0],file[jaw()]['vtk'],args.add_inname[0])


        surf_input = ReadSurf(file[jaw.inv()]['vtk'])
        surf_output = TransformSurf(surf_input,output_icp['matrix'])
        WriteSurf(surf_output,args.output_folder[0],file[jaw.inv()]['vtk'],args.add_inname[0])

        json_input = LoadJsonLandmarks(file[jaw.inv()]['json'])
        json_output = ApplyTransform(json_input,output_icp['matrix'])
        WriteJsonLandmarks(json_output,file[jaw.inv()]['json'],file[jaw.inv()]['json'],args.add_inname[0],args.output_folder[0])

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
    parser.add_argument('list_landmark',nargs=1)
    parser.add_argument('jaw',nargs=1)

    args = parser.parse_args()


    main(args)