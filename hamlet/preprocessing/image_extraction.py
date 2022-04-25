"""Batch converts the images in DICOM files to .png."""
import numpy as np
import scipy as sp
import argparse
import cv2
import os

from pydicom import dcmread
from matplotlib import pyplot as plt
from multiprocessing import Pool
from skimage import transform, exposure
from skimage.color import rgb2gray
from PIL import Image, ImageChops
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut


def good_dicom(ds):
    """Determines if a DICOM file has a usable image.
    
      Parameters 
      ----------
        ds : a Pydicom dataset as returned by pydicom.dcmread()
      
      Returns
      ----------
        True if the dataset meets certain conditions, else False.
    """
    bad_id = False
    bad_image = False
    no_tsid = 'TransferSyntaxUID' not in ds.file_meta
    if not no_tsid:
        tsid = ds.file_meta.TransferSyntaxUID
        if tsid in ['1.2.840.10008.1.2.4.53', '1.2.840.10008.1.2.4.55']:
            bad_id = True
    no_pixels = ('PixelData' not in ds or 'PhotometricInterpretation' not in ds)
    try:
        ds.pixel_array
    except:
        bad_pixels = True
    
    if np.any([no_tsid, bad_id, no_pixels, bad_pixels]):
        return False
    else:
        return True


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


def convert_to_png(file, 
                   file_dir, 
                   img_dir,
                   prefix=None,
                   shape=1024,
                   use_modality_lut=False,
                   use_voi_lut=True):
    """Extracts the image in a DICOM file to .png.
      
    Parameters
    ----------
    file : str
        The name of the input dicom file, ending in ".png".
    file_dir : str
        The path to the directory holding the dicom file.
    img_dir : str
        The path to the directory where the image should be exported.
    prefix : str, default=None
        Optional prefix for the exported image's file name.
    shape : int, default=1024
        Width in pixels of the image after export. For no rescaling, set to 
        None.
    use_modality_lut : bool, default=False
        Whether to apply the modality LUT from the DICOM file to the image
    use_voi_lut : bool, default=True
        Whether to apply the VOI LUT from the DICOM file to the image
    
    Returns
    ----------
    Nothing (the image is written to the output directory).
    """
    # Read in the DICOM file
    ds = dcmread(file_dir + file, force=True)
    
    if not good_dicom(ds):
        return
    else:
        # Convert to float to avoid overflow or underflow losses.
        image = ds.pixel_array
        
        # Converting RGB
        if (ds.PhotometricInterpretation == 'RGB' or
            ds.SamplesPerPixel == 3):
            image = rgb2gray(image)
        
        # Applying any windowing
        if use_modality_lut:
            try:
                image = apply_modality_lut(image, ds)
            except:
                pass
        if use_voi_lut:
            try:
                image = apply_voi_lut(image, ds)
            except:
                pass
        
        # Exposure adjustment and rescaling grey scale between 0-255
        image = rescale(image)
        
        # Trimming black or white borders
        pil_image = Image.fromarray(image)
        image = np.asarray(trim(pil_image))
            
        # Optional resize
        if shape:
            arr_shape = image.shape
            y = shape
            x = int((shape / arr_shape[1]) * arr_shape[0])
            shape = (x, y)
            image = transform.resize(image, shape)
        
        # Convert to grayscale and write to disk    
        cm = plt.get_cmap('gray')
        image = np.array(cm(image) * 255).astype(np.uint8)
        
        # Check for inversion, if known
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            if len(image.shape) < 3:
                image = np.invert(image)
            else:
                image = np.invert(image[:, :, :-1])
        
        # Final check for images that are too bright or dark
        if not good_brightness(img):
            print(file)
            print('Image too bright or too dark')
            return
        else:
            png_name = file.replace('dcm', 'png')
            if prefix:
                png_name = prefix + png_name
            
            cv2.imwrite(img_dir + png_name, image)
        return


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
        input = [(f, DICOM_DIR, IMG_DIR, PREFIX, 1024) for f in files]
        output = p.starmap(convert_to_png, input)
        p.close()
        p.join()
    