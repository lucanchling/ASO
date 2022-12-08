#!/usr/bin/env python-real

import sys
import os
import glob
import time
import argparse
import numpy as np
import SimpleITK as sitk
from utils.utils import WriteJsonLandmarks,ICP

def WriteTXT(text,file='/home/luciacev/Desktop/Luc_Anchling/Projects/ASO/ASO_CBCT/sumup.txt'):
    with open(file,'a') as f:
        f.write(str(text)+'\n')


def main(args):
    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    lm_extension = ['.json']

    list_landmark = args.list_landmark[0].split(',')[:-1]
    input_dir, gold_dir, out_dir = args.input[0], args.gold_folder[0], args.output_folder[0]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    normpath = os.path.normpath("/".join([gold_dir, '**', '']))
    for file in glob.iglob(normpath, recursive=True):
        if os.path.isfile(file) and True in [ext in file for ext in lm_extension]:
            gold_json_file = file
        if os.path.isfile(file) and True in [ext in file for ext in scan_extension]:
            gold_file = file
    
    input_files = []
    input_json_files = []
    normpath = os.path.normpath("/".join([input_dir, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(file) and True in [ext in file for ext in lm_extension]:
            input_json_files.append(file)
        if os.path.isfile(file) and True in [ext in file for ext in scan_extension]:
            input_files.append(file)
    
    for i in range(len(input_files)):
        input_file,input_json_file = input_files[i],input_json_files[i]

        output,source_transformed = ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark)
        
        # Write JSON
        dir_json = os.path.dirname(input_json_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)
        
        json_path = os.path.join(dir_json,os.path.basename(input_json_file).split('.mrk.json')[0]+'_'+args.add_inname[0]+'.mrk.json')

        WriteJsonLandmarks(source_transformed, input_json_file, output_file=json_path)

        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,os.path.basename(input_file).split('.')[0]+'_'+args.add_inname[0]+'.nii.gz')
        sitk.WriteImage(output, file_outpath)


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

    args = parser.parse_args()

    main(args)
