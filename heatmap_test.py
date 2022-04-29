import numpy as np
import tensorflow as tf
import PIL.Image
import saliency.core as saliency
import os

from matplotlib import pyplot as plt

from hamlet import attribution
from hamlet.tools import image as tim
from hamlet.modeling import models


# Loading some images
im_dir = 'img/test/'
im_files = os.listdir(im_dir)
im_paths = [im_dir + s for s in im_files]
im = tim.load_image(im_paths[0], (600, 600))

# Loading a model
model = models.EfficientNet(num_classes=1, augmentation=False)
model.load_weights('output/binary/checkpoints/training/')
conv_layer = model.get_layer('top_conv')
model = tf.keras.models.Model([model.inputs], 
                              [conv_layer.output, model.output])
call_model_args = {'class_id': 0,
                   'model': model}

# Computing some masks
masks = attribution.compute_masks(image=im,
                                  methods=['ig'],
                                  xrai_mode='full',
                                  call_model_args=call_model_args,
                                  batch_size=1)

# Plotting
attribution.panel_plot(im, masks, 'Gradient')
