#!/usr/bin/env python3
# Convert audio files to a fixed sample rate (22050Hz) mp3 using ffmpeg

import shutil
import os
import glob
import subprocess
from tqdm import tqdm
import itertools
import concurrent.futures
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='source dir')
    parser.add_argument('dst', help='target dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    SAMPLE_RATE = 22050

    args = get_arguments()
    src = args.src
    dst = args.dst
    print(f'{src} --> {dst}')

    os.makedirs(dst, exist_ok=True)
    subdirs = os.listdir(src)
    for subdir in subdirs:
        input_folder = os.path.join(src, subdir)
        input_files = os.listdir(input_folder)
        print(f'{len(input_files)} files under {input_folder}')
        output_folder = os.path.join(dst, subdir)
        os.makedirs(output_folder, exist_ok=True)
        jobs = []
        for input_file in input_files:
            if input_file.endswith('.mp3'):
                output_file = input_file
            else:
                output_file = input_file[:input_file.rfind('.')] + '.mp3'
                print(f'{input_file} --> {output_file}')
            input_path = os.path.join(src, subdir, input_file)
            output_path = os.path.join(dst, subdir, output_file)
            jobs.append(['ffmpeg', '-i', input_path, '-ar', f'{SAMPLE_RATE}', output_path])
        # Run the jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            res = [executor.submit(subprocess.run, job, check=True) for job in jobs]
            for r in tqdm(concurrent.futures.as_completed(res), total=len(jobs)):
                pass

    

    