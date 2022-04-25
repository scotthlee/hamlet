"""Makes Grad-CAM heatmaps for x-rays.

Mapping functions from Keras tutorial at 
https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow import keras
from IPython.display import Image, display
from multiprocessing import Pool


def get_img_array(img_path, size):
    """Loads and image from path and returns it as an array."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    return array


def make_gradcam_heatmap(img_array, 
                         model, 
                         last_conv_layer_name, 
                         pred_index=None):
    """Makes a Grad-CAM heatmap from an image array and a trained model."""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, 
                         model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap
    # between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(img_array,
                    heatmap,
                    alpha=0.4,
                    scale=True,
                    pixel_type='float',
                    return_image=False):
    """Lays a Grad-CAM heatmap over the original image."""
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], 
                                      img_array.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    supe_img = jet_heatmap * alpha + img_array
    if scale:
        supe_img = supe_img / supe_img.max()
    if pixel_type != 'float':
        supe_img = np.round(supe_img * 255).astype(np.uint8)
    if return_image:
        supe_img = tf.keras.preprocessing.image.array_to_img(supe_img)
    
    return supe_img


def write_gradcam(img_file, 
                  img_dir, 
                  out_dir,
                  model,
                  conv_layer,
                  pred_index=None,
                  img_width=600,
                  img_height=600,
                  write_original=False):
    img_name = img_file[:-4]
    img_path = img_dir + img_file
    img_array = get_img_array(img_path,
                              size=(img_width, img_height))
    img_array_exp = np.expand_dims(img_array, axis=0)
    heatmap = make_gradcam_heatmap(img_array_exp,
                                   model,
                                   conv_layer,
                                   pred_index)
    overlay = overlay_gradcam(img_array,
                              heatmap,
                              pixel_type='float',
                              alpha=0.4)
    plt.imsave(out_dir + img_name + '_over.png', overlay)
    if write_original:
        plt.imsave(out_dir + img_file, img_array / 255)
    
    return

        