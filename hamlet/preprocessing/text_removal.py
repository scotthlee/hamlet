"""Filters out x-rays with burned-in text from the main dataset."""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import pytesseract

from pytesseract import Output
from multiprocessing import Pool


def check_text(file_path, lim=3):
    """Checks an image for burned-in text.
    
    Parameters
    ----------
    file_path : str
        Full path to the image file to be checked.
    lim : int, default=3
        Maximum number of words an image is allowed to have before being 
        flagged.
    
    Returns
    ----------
    True if the image has more than lim words, else False.    
    """
    img = cv2.imread(file_path)
    tess = pytesseract.image_to_data(img, output_type=Output.DICT)
    words = tess['text']
    if np.any([len(w) > lim for w in words]):
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str,
                        default='D:/data/hamlet/source/immigrant/',
                        help='path to the directory holding the images')
    parser.add_argument('--text_dir',
                        type=str,
                        default='D:/data/hamlet/source/bad/text/',
                        help='path to the directory for dumping the images \
                        with too much text')
    parser.add_argument('--num_words',
                        type=int,
                        default=3,
                        help='maximum allowable number of words per image')
    args = parser.parse_args()
    
    # Setting globals
    IMG_DIR = args.img_dir
    TEXT_DIR = args.text_dir
    NUM_WORDS = args.num_words
    
    # Importing the data
    files = os.listdir(IMG_DIR)
    
    # Checking the files
    with Pool() as p:
        input = [(IMG_DIR + f, NUM_WORDS) for f in files]
        res1 = p.starmap(check_text, input)
        p.close()
        p.join()
    
    # Moving the files with text
    with_text = np.where(res1)[0]
    to_move = [files[i] for i in with_text]
    with Pool() as p:
        input = [(IMG_DIR + f, TEXT_DIR + f) for f in to_move]
        res2 = p.starmap(os.rename, input)
        p.close()
        p.join()
