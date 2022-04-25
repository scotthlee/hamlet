"""Makes Grad-CAM heatmaps for x-rays.

Mapping functions from Keras tutorial at 
https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow import keras
from multiprocessing import Pool

from modeling import models
from attribution.grad_cam import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str,
                        help='Path to the image files.')
    parser.add_argument('--out_dir',
                        type=str,
                        default=None,
                        help='Path for saving the heatmaps.')
    parser.add_argument('--mod_dir',
                        type=str,
                        default='../output/binary/checkpoints/training/',
                        help='Path to the model checkpoint.')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1,
                        help='Number of output classes. Must match the size \
                        of the top layer in the model from mod_dir.')
    parser.add_argument('--augment',
                        action='store_true')
    parser.add_argument('--multi_type',
                        type=str,
                        default='label',
                        help='Kind of top layer to use for multiclass problems.\
                        Only applies when --num_classes > 2.',
                        choices=['label', 'class'])
    parser.add_argument('--img_width',
                        type=int,
                        default=600,
                        help='How wide the model expects the image to be. Must \
                        match the dims for the checkpoint in --mod_dir.')
    parser.set_defaults(augment=False)
    args = parser.parse_args()
    
    IMG_DIR = args.img_dir
    MOD_DIR = args.mod_dir
    NUM_CLASSES = args.num_classes
    AUGMENT = args.augment
    MULTI_TYPE = args.multi_type
    IMG_HEIGHT, IMG_WIDTH = args.img_width, args.img_width
    
    if args.out_dir:
            OUT_DIR = args.out_dir
    else:
        OUT_DIR = IMG_DIR
    
    # Loading the trained model
    model = models.EfficientNet(num_classes=NUM_CLASSES,
                        multi_type=MULTI_TYPE,
                        img_height=IMG_HEIGHT,
                        img_width=IMG_WIDTH,
                        augmentation=AUGMENT)
    model.load_weights(MOD_DIR)
    model.layers[-1].activation = None
    last_conv_layer_name = 'top_conv'
    
    # Running the heatmaps
    files_to_map = os.listdir(IMG_DIR)
    for f in files_to_map:
        write_gradcam(img_file=f,
                      img_dir=IMG_DIR, 
                      out_dir=OUT_DIR, 
                      model=model,
                      conv_layer=last_conv_layer_name)
