"""Batch converts the images in DICOM files to .png."""
import numpy as np
import argparse
import os

from multiprocessing import Pool

from hamlet.tools.dicom import convert_to_png


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir',
                        type=str,
                        default='X:/DICOMM/Immigrant/',
                        help='directory holding the DICOM files')
    parser.add_argument('--img_dir',
                        type=str,
                        default='D:/data/hamlet/source/immigrant/',
                        help='output directory for the image files')
    parser.add_argument('--prefix',
                        type=str,
                        default='im_',
                        help='prefix for the image file names to identify \
                        which dataset they came from')
    parser.add_argument('--num_files',
                        type=int,
                        default=-1,
                        help='limit on number of files to process')
    parser.add_argument('--overwrite',
                        action='store_true')
    parser.add_argument('--convert_PR',
                        action='store_true')
    parser.add_argument('--processes',
                        type=int,
                        default=-1,
                        help='number of processes for the Pool')
    parser.set_defaults(overwrite=False,
                        convert_PR=False)
    args = parser.parse_args()

    PROCESSES = args.processes if args.processes != -1 else None
    CONVERT_PR = args.convert_PR
    NUM_FILES = args.num_files
    DICOM_DIR = args.dicom_dir
    IMG_DIR = args.img_dir
    PREFIX = args.prefix
    OVERWRITE = args.overwrite

    # Making the list of files; default is to convert new ones only
    to_convert = [f for f in os.listdir(DICOM_DIR) if 'dcm' in f]

    if not CONVERT_PR:
        to_convert = [f for f in to_convert if '_PR' not in f]

    if not OVERWRITE:
        img_files = [f for f in os.listdir(IMG_DIR)]
        img_files = [f.replace('png', 'dcm') for f in img_files]
        to_convert = np.setdiff1d(to_convert, img_files)

    if NUM_FILES == -1:
        NUM_FILES = len(to_convert)

    files = [f for f in to_convert][:NUM_FILES]
    print(len(files))

    with Pool(PROCESSES) as p:
        input = [(f, DICOM_DIR, IMG_DIR,
                  PREFIX, 1024, False, 'X:/DICOMM/bad_files.csv')
                 for f in files]
        output = p.starmap(convert_to_png, input)
        p.close()
        p.join()
