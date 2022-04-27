import numpy as np
import tensorflow as tf
import os
import cv2
import pytesseract

from pytessearact import Output
from skimage import transform, exposure
from skimage.color import rgb2gray
from PIL import Image, ImageChops


def good_brightness(img,
                    min=50,
                    max=215):
    """Checks if an image is either too bright or too dark."""
    if int(img.mean()) not in np.arange(min, max):
        return False
    else:
        return True


def trim(im):
    """Trims a solid border from an image.
    
    Parameters
    ----------
    im : NumPy array
        Image to be trimmed.
    
    Returns
    ----------
    The image without the border.
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im


def rescale(image, new_max=255):
    """Rescales pixel values so they fall between 0 and a new max.
    
    Parameters
    ----------
    image : array
        The pixel array to be rescaled.
    new_max : int, default=255
        The maximum pixel value for the array after rescaling.
    
    Returns
    ----------
    The rescaled image as a NumPy array.
    """
    adjusted = np.array(np.maximum(image, 0) / image.max())
    scaled =  np.array(adjusted * new_max, dtype=np.uint8)
    return scaled


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


def check_valid_image(files, print_fname=True):
    """Checks a list of image files to make sure they're valid."""
    for f in f:
        if print_fname:
            print(f)
        image = tf.io.read_file(cxr_dir + f)
        if 'png' in f:
            image = tf.io.decode_png(image)
        elif 'jpeg' in f:
            image = tf.io.decode_jpeg(image)
        else:
            print('Images should be either PNG or JPEG')
    return


def flip_image(img_name, img_dir):
    """Opens an image, inverts it, and overwrites the original file 
    with the new version.
    """
    img = cv2.imread(IMG_DIR + img_name)
    inv_img = np.invert(img)
    cv2.imwrite(IMG_DIR + img_name, inv_img)
    return


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

