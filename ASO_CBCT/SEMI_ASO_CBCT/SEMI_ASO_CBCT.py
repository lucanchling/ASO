#!/usr/bin/env python-real

import sys
import os
import glob
import time
import argparse
import numpy as np
import SimpleITK as sitk

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from utils import WriteJsonLandmarks,ICP,ExtractFilesFromFolder,MergeJson,WriteJson

def main(args):
    
    if not os.path.exists(os.path.split(args.log_path[0])[0]):
        os.mkdir(os.path.split(args.log_path[0])[0])

    with open(args.log_path[0],'w') as log_f :
        log_f.truncate(0)
    
    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    lm_extension = ['.json']

    list_landmark = args.list_landmark[0].split(' ')
    input_dir, gold_dir, out_dir = args.input[0], args.gold_folder[0], args.output_folder[0]
    
    MergeJson(input_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    input_files, input_json_files = ExtractFilesFromFolder(input_dir, scan_extension, lm_extension)

    gold_file, gold_json_file = ExtractFilesFromFolder(gold_dir, scan_extension, lm_extension, gold=True)
    
    for i in range(len(input_files)):
        
        input_file, input_json_file = input_files[i],input_json_files[i]

        output, source_transformed = ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark)
        
        # Write JSON
        dir_json = os.path.dirname(input_json_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)
        
        json_path = os.path.join(dir_json,os.path.basename(input_json_file).split('.mrk.json')[0]+'_'+args.add_inname[0]+'.mrk.json')

        if not os.path.exists(json_path):
            WriteJson(source_transformed,json_path)

        #WriteJsonLandmarks(source_transformed, input_json_file, output_file=json_path)

        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,os.path.basename(input_file).split('.')[0]+'_'+args.add_inname[0]+'.nii.gz')
        if not os.path.exists(file_outpath):
            sitk.WriteImage(output, file_outpath)


        with open(args.log_path[0],'r+') as log_f:
            log_f.write(str(i))

if __name__ == "__main__":
    
    print("Starting")
    print(sys.argv)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_landmark',nargs=1)
    parser.add_argument('log_path',nargs=1)
    
    args = parser.parse_args()

    main(args)