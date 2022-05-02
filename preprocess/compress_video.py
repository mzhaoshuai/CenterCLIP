# coding=utf-8
"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import os
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def compress(paras):
    input_video_path, output_video_path = paras
    try:
        # https://trac.ffmpeg.org/ticket/309
        command = ['ffmpeg',
                   '-y',                    # (optional) overwrite output file if it exists
                   '-i', input_video_path,
                   '-filter:v',
                   'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
                   '-map', '0:v',
                   '-r', '3',              # frames per second
                   output_video_path,
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def prepare_input_output_pairs(input_root, output_root, ignore_exist=True):
    """get input and ouput video pairs"""
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            if os.path.exists(output_video_path) and ignore_exist:
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path)
    
    return input_video_path_list, output_video_path_list

# os.path.expanduser('~/dataset1/msvd/resized_video_3fps')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str,
                            default='/home/linchao/Downloads/DiDeMo',
                            help='input root')
    parser.add_argument('--output_root', type=str,
                            default='/home/linchao/Downloads/DiDeMo_resized',
                            help='output root')
    parser.add_argument('--num_works', type=str, default=16,
                            help='the number of processes')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    num_works = args.num_works

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)

    print("Total video need to process: {}".format(len(input_video_path_list)))
    print("Begin with {}-core logical processor.".format(num_works))

    # process pool
    pool = Pool(num_works)
    for _ in tqdm(pool.imap(compress,
                              [(input_video_path, output_video_path) for
                               input_video_path, output_video_path in
                               zip(input_video_path_list, output_video_path_list)]),
                    total=len(input_video_path_list)):
        pass

    pool.close()
    pool.join()
