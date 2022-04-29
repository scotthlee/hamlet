import numpy as np
import tensorflow as tf
import PIL.Image
import saliency.core as saliency
import os

from matplotlib import pyplot as plt

from tools import image as tim
from modeling import models


def call_model(images, call_model_args=None, expected_keys=None):
    """Generic function for getting predictions and gradients from a model."""
    model = call_model_args['model']
    target_class_idx =  call_model_args['class_id']
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            _, output_layer = model(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}


def compute_masks(image,
                  model,
                  call_model_args,
                  smooth=True,
                  methods='all',
                  xrai_method='full',
                  batch_size=20):
    method_dict = {
        'grad': 'GradientSaliency',
        'blur': 'BlurIG',
        'ig': 'IntegratedGradients',
        'gig': 'GuidedIG',
        'gradcam': 'GradCam',
        'xrai': 'XRAI'
    }
    if methods == 'all':
        obj_names = list(method_dict.values())
    else:
        obj_names = [method_dict[m] for m in methods]
    
    all_masks = []
    # Masks for algorithms that don't compute gradients along a path
    for obj_name in obj_names:
        mess = 'Computing masks for ' + obj_name
        print(mess)
        obj = getattr(saliency, obj_name)()
        if obj_name in ['GradientSaliency', 'GradCam']:
            obj_masks = [obj.GetMask(image,
                                     call_model,
                                     call_model_args)]
            if smooth:
                obj_masks += [obj.GetSmoothedMask(image,
                                                  call_model,
                                                  call_model_args)]
        else:
            # Masks for the algorithms that compute gradients along a path (and 
            # thus take the batch_size parameter to manage memory)  
            if obj_name == 'XRAI':
                # Separate step for XRAI to allow for full vs. fast versions of 
                # the algorithm  
                xrp = saliency.XRAIParameters()
                xrp.algorithm = xrai_method
                obj_masks = [obj.GetMask(image,
                                         call_model,
                                         call_model_args,
                                         extra_parameters=xrp,
                                         batch_size=batch_size)]
                # All-white mask since XRAI doesn't do smoothing
                obj_masks += np.zeros(obj_masks[0].shape,) + 1.0
            else:
                obj_masks = [obj.GetMask(image, 
                                         call_model, 
                                         call_model_args,
                                         batch_size=batch_size)]
                if smooth:
                    obj_masks += [obj.GetSmoothedMask(image,
                                                      call_model,
                                                      call_model_args,
                                                      batch_size=batch_size)]
        
        gray_masks = [saliency.VisualizeImageGrayscale(m) for m in obj_masks]
        all_masks += gray_masks
    
    return all_masks


def panel_plot(image, 
               masks, 
               method_name):
    # Set up matplot lib figures.
    ROWS = 1
    COLS = 5
    UPSCALE_FACTOR = 20
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    
    # Fill hte panels
    tim.show_grayscale_image(image / 255,
                             title='Original',
                             ax=plt.subplot(ROWS, COLS, 1))
    tim.show_grayscale_image(masks[0], 
                             title=method, 
                             ax=plt.subplot(ROWS, COLS, 2))
    tim.show_grayscale_image(masks[1], 
                             title=method + ' (smooth)', 
                             ax=plt.subplot(ROWS, COLS, 3))
    tim.show_heatmap(tim.overlay_heatmap(image, masks[0]),
               title='Overlay',
               ax=plt.subplot(ROWS, COLS, 4))
    tim.show_image(tim.overlay_heatmap(image, masks[1]),
                     title='Overlay (Smooth)',
                     ax=plt.subplot(ROWS, COLS, 5))
    plt.tight_layout()
    plt.show()

