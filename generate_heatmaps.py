"""Writes a report with the models' predictions for a set of new images.

Notes:
  1. 'img_dir' points to the directory holding the folder that holds the image
    files for prediction. That folder should be named 'img/'.

"""
import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
import saliency.core as saliency

from hamlet import attribution, models
from hamlet.tools import image as tim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        help='Path to the folder holding the images for  \
                        prediction.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Path where the script should dump the output \
                        files. Writes to img_dir by default.')
    parser.add_argument('--method',
                        type=str,
                        default='xrai',
                        help='Attribution method to use.',
                        choices=['gradient', 'blur', 'gig',
                                 'ig', 'gradcam', 'xrai'])
    parser.add_argument('--mod_dir',
                        type=str,
                        default='output/abnormal/checkpoints/training/',
                        help='Path to the folder holding the trained \
                        model.')
    parser.add_argument('--model_flavor',
                        type=str,
                        default='EfficientNetV2M',
                        help='What pretrained model to use as the feature \
                        extractor.')
    parser.add_argument('--no_augmentation',
                        action='store_true',
                        help='Specifies that the models should be built with \
                        the image augmentation layer.')
    parser.add_argument('--image_dim',
                        type=int,
                        default=600,
                        help='Either dimension of the image to be passed \
                        to the model.')
    parser.add_argument('--single_GPU',
                        action='store_true',
                        help='Turns off distributed (multi-GPU) training')
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Prefix for the predictions file.')
    parser.add_argument('--scale',
                        type=float,
                        default=10.0,
                        help='Scaling parameter for figure size.')
    parser.set_defaults(no_augmentation=False,
                        single_GPU=False)
    args = parser.parse_args()

    # Setting things up
    AUGMENT = not args.no_augmentation
    METHOD = args.method
    MOD_DIR = args.mod_dir
    MODEL_FLAVOR = args.model_flavor
    IMG_DIR = args.image_dir
    IMG_DIM = args.image_dim
    OUT_DIR = IMG_DIR
    DISTRIBUTED = not args.single_GPU
    PREFIX = args.prefix
    SCALE = args.scale
    if args.output_dir is not None:
        OUT_DIR = args.output_dir

# Setting training strategy
if DISTRIBUTED:
    print('Using multiple GPUs.\n')
    cdo = tf.distribute.HierarchicalCopyAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
else:
    strategy = tf.distribute.get_strategy()

# Loading the data
test_files = os.listdir(IMG_DIR)
test_ids = [f[:-4] for f in test_files]
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  IMG_DIR,
  labels=None,
  shuffle=False,
  image_size=(IMG_DIM, IMG_DIM)
)

# Loading the images
im_files = os.listdir(IMG_DIR)
im_paths = [IMG_DIR + s for s in im_files]


# Loading the trained model
with strategy.scope():
    mod = models.EfficientNet(num_classes=1,
                              img_height=IMG_DIM,
                              img_width=IMG_DIM,
                              augmentation=AUGMENT,
                              model_flavor=MODEL_FLAVOR)
    mod.load_weights(MOD_DIR)
    conv_layer = mod.get_layer('top_conv')
    mod = tf.keras.models.Model([mod.inputs],
                                [conv_layer.output, mod.output])
    call_model_args = {'class_id': 0,
                       'model': mod}

    # Trying the single-image plot
    for i, im_path in enumerate(im_paths):
        print('Making a ' + METHOD + ' heatmap for ' + im_files[i])
        im = tim.load_image(im_path, (IMG_DIM, IMG_DIM))
        mask, method_name = attribution.compute_masks(
                                        image=im,
                                        methods=[METHOD],
                                        call_model_args=call_model_args,
                                        batch_size=1)
        attribution.panel_plot(images=[im],
                               masks=mask,
                               method_name=method_name[0],
                               save_dir=OUT_DIR,
                               image_id=im_files[i][:-4],
                               show=False,
                               save=True,
                               scale=SCALE)
