"""Classes and functions for prepping the DICOM files"""

import numpy as np
import pandas as pd
import scipy as sp
import cv2
import pytesseract
import os

from pytesseract import Output
from multiprocessing import Pool
from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from pydicom import dcmread
from skimage import transform, exposure
from skimage.color import rgb2gray
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_windowing


def image_augmentation(image,
                       contrast=(0.5, 2.0),
                       hue=0.1,
                       saturation=(0.7, 1.25),
                       brightness=0.2):
    """ Randomly alters the contrast, hue, saturation, and brightness of an
    input image. Mostly used for manually experimenting with augmentation
    values. For modeling, please use the Augmentation layer.
    
    Parameters
    ----------
    flip : bool, default=True
        Whether to apply random left-right flips during augmentation.
    hue : float, default=0.1
        Random factor for adjusting hue during augmentation.
    contrast : tuple of floats, default=(0.5, 2)
        Lower and upper bounds for the random contrast factor to be applied
        during augmentation.
    saturation : tuple of floats, default=(0.7, 1.25)
        Lower and upper bounds for the random saturation factor to be 
        applied during augmentation.
    brightness : float, default=0.2
        Max random factor for brightness adjustment during augmentation. 
        Must be greater than 0.
    """
    if flip:
        image = tf.image.random_flip_left_right(image)
    
    image = tf.image.random_contrast(image, contrast[0], contrast[1])
    image = tf.image.random_hue(image, hue)
    image = tf.image.random_saturation(image, saturation[0], saturation[1])
    image = tf.image.random_brightness(image, brightness) 
    return image


def check_fname(fname, ids):
    """Checks a list of IDs for a single file name."""
    return fname in ids


def check_fnames(fnames, ids):
    """Checks a list of IDs for a list of file names."""
    with Pool() as p:
        input = [(f, ids) for f in fnames]
        res = p.starmap(check_fname, input)
        p.close()
        p.join()
    
    return np.array(res)


def check_valid_image(files):
    """Checks a list of image files to make sure they're valid."""
    for f in f:
        print(f)
        image = tf.io.read_file(cxr_dir + f)
        if 'png' in f:
            image = tf.io.decode_png(image)
        else:
            image = tf.io.decode_jpeg(image)


def trim_zeroes(fname):
    """Removes extra 0s from panel file names."""
    cut = np.where([s == '_' for s in fname])[0][1] + 1
    ending = fname[cut:]
    if len(ending) == 6:
        return fname
    else:
        drop = len(ending) - 6
        ending = ending[drop:]
        return fname[:cut] + ending


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    return total_size
