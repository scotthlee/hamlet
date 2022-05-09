import numpy as np
import tensorflow as tf
import pickle
import saliency.core as saliency
import os

from matplotlib import pyplot as plt

from hamlet import attribution, models
from hamlet.tools import image as tim


# Loading some images
im_dir = 'img/test/'
im_files = os.listdir(im_dir)
im_paths = [im_dir + s for s in im_files]
im = tim.load_image(im_paths[1], (600, 600))

# Loading a model
model = models.EfficientNet(num_classes=1, augmentation=False)
model.load_weights('output/binary/checkpoints/training/')
conv_layer = model.get_layer('top_conv')
model = tf.keras.models.Model([model.inputs], 
                              [conv_layer.output, model.output])
call_model_args = {'class_id': 0,
                   'model': model}

# Computing some masks
test_masks, test_methods = attribution.compute_masks(image=im,
                                                     methods='all',
                                           call_model_args=call_model_args,
                                           batch_size=1)
pickle.dump(test_masks, open('img/masks.pkl', 'wb'))

# Plotting
attribution.panel_plot(image=im, 
                       masks=test_masks, 
                       methods=test_methods)
