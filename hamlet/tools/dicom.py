import numpy as np
import scipy as sp
import pandas as pd
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

from .image import rescale, trim, good_brightness


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
    bad_pixels = False

    # Checking the transfer syntax ID
    no_tsid = 'TransferSyntaxUID' not in ds.file_meta
    if not no_tsid:
        tsid = ds.file_meta.TransferSyntaxUID
        if tsid in ['1.2.840.10008.1.2.4.53', '1.2.840.10008.1.2.4.55']:
            bad_id = True

    # Checking the pixel data
    no_pixels = ('PixelData' not in ds or 'PhotometricInterpretation' not in ds)
    if not no_pixels:
        try:
            ds.pixel_array
        except:
            bad_pixels = True

    # Gathering the info
    conditions = [no_tsid, bad_id, no_pixels, bad_pixels]
    if np.any(conditions):
        condition_ints = np.array(conditions, dtype=np.uint8)
        condition_ints = [n for n in condition_ints]
        return (False, condition_ints)
    else:
        return (True, '')


def convert_to_png(file,
                   file_dir,
                   img_dir,
                   prefix=None,
                   shape=1024,
                   write_image=True,
                   error_report=True,
                   error_report_filename='bad_files.csv',
                   er_dir=None,
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
    error_report : bool, default=True
        Whether to document image extraction errors in a CSV file.
    error_report_filename : str, default='bad_files.csv'
        Name for the error report, if generated.
    er_dir : str, default=None
        Where to save the error report. Unless this is specified, the report 
        will be saved to img_dir.
    use_modality_lut : bool, default=False
        Whether to apply the modality LUT from the DICOM file to the image
    use_voi_lut : bool, default=True
        Whether to apply the VOI LUT from the DICOM file to the image

    Returns
    ----------
    Nothing (the image is written to the output directory).
    """
    # Read in the DICOM file and check it for basic problems
    ds = dcmread(file_dir + file, force=True)
    good_file, errors = good_dicom(ds)
    
    if not good_file:
        if error_report:
            if not er_dir:
                er_dir = img_dir
            efn = error_report_filename
            report = [prefix, file] + errors + [0, 0]
            pd.DataFrame(report).transpose().to_csv(er_dir + efn,
                                                    mode='a',
                                                    index=False,
                                                    header=False)
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
            if bff:
                report = [prefix, file] + [0, 0, 0, 0] + [1, 0]
                pd.DataFrame(report).transpose().to_csv(bff,
                                                        mode='a',
                                                        index=False,
                                                        header=False)

        # Final check for images that are too bright or dark
        if not good_brightness(image):
            report = [prefix, file] + [0, 0, 0, 0] + [0, 1]
            pd.DataFrame(report).transpose().to_csv(bff,
                                                    mode='a',
                                                    index=False,
                                                    header=False)
            return
        else:
            if write_image:
                if 'dcm' in file:
                    png_name = file.replace('dcm', 'png')
                else:
                    png_name = file.replace('dicom', 'png')
                if prefix:
                    png_name = prefix + png_name

                cv2.imwrite(img_dir + png_name, image)
        return