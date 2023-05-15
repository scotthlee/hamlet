"""Batch converts the images in DICOM files to .png."""
import numpy as np
import pandas as pd
import argparse
import os

from multiprocessing import Pool

from hamlet.tools.dicom import convert_to_png


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir',
                        type=str,
                        default='X:/DICOMM/Immigrant/20220201/',
                        help='Directory holding the DICOM files.')
    parser.add_argument('--img_dir',
                        type=str,
                        default='D:/data/hamlet/source/immigrant/',
                        help='Output directory for the image files.')
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Prefix for the image file names to identify \
                        which dataset they came from.')
    parser.add_argument('--img_dim',
                        type=int,
                        default=1024,
                        help='Desired height or width of the image file.')
    parser.add_argument('--num_files',
                        type=int,
                        default=-1,
                        help='Limit on number of files to process.')
    parser.add_argument('--error_report_filename',
                        type=str,
                        default='bad_files.csv',
                        help='Name for the optional error report.')
    parser.add_argument('--error_report_dir',
                        type=str,
                        default=None,
                        help='Where the error report should be saved.')
    parser.add_argument('--overwrite',
                        action='store_true')
    parser.add_argument('--no_error_report',
                        action='store_true')
    parser.add_argument('--convert_PR',
                        action='store_true')
    parser.add_argument('--processes',
                        type=int,
                        default=-1,
                        help='number of processes for the Pool')
    parser.set_defaults(overwrite=False,
                        convert_PR=False,
                        no_error_report=False)
    args = parser.parse_args()

    PROCESSES = args.processes if args.processes != -1 else None
    CONVERT_PR = args.convert_PR
    NUM_FILES = args.num_files
    DICOM_DIR = args.dicom_dir
    IMG_DIR = args.img_dir
    IMG_DIM = args.img_dim
    PREFIX = args.prefix
    OVERWRITE = args.overwrite
    ERROR_REPORT = not args.no_error_report
    EFN = args.error_report_filename
    ER_DIR = IMG_DIR if not args.error_report_dir else args.error_report_dir

    # Making the list of files; default is to convert new ones only
    to_convert = [f for f in os.listdir(DICOM_DIR) if ('dcm' in f)
                  or ('dicom' in f)]

    # Option to convert presentation state files
    if not CONVERT_PR:
        to_convert = [f for f in to_convert if '_PR' not in f]

    # Option to overwrite images that have already been extracted
    if not OVERWRITE:
        img_files = [f for f in os.listdir(IMG_DIR)]
        img_files = [f.replace('png', 'dcm') for f in img_files]
        to_convert = np.setdiff1d(to_convert, img_files)

    # Setting the number of files to extract
    if NUM_FILES == -1:
        NUM_FILES = len(to_convert)

    # Making an empty error report file to write to, if one doesn't exist
    if ERROR_REPORT:
        if EFN not in os.listdir(ER_DIR):
            report = [''] * 8
            report_df = pd.DataFrame(report).transpose()
            report_df.columns = ['group', 'file_name',	'no_TS_ID',
                                 'unsupported_TS_ID', 'missing_pixels',
                                 'corrupt_pixels', 'inverted_colors',
                                 'brightness_issues']
            report_df.to_csv(ER_DIR + EFN, index=False)

    files = [f for f in to_convert][:NUM_FILES]
    print(len(files))

    with Pool(PROCESSES) as p:
        input = [(f, DICOM_DIR, IMG_DIR,
                  PREFIX, IMG_DIM, True,
                  ERROR_REPORT, EFN, ER_DIR)
                 for f in files]
        output = p.starmap(convert_to_png, input)
        p.close()
        p.join()