import numpy as np
import tensorflow as tf
import PIL.Image
import saliency.core as saliency
import numba
import os

from matplotlib import pyplot as plt
from numba import cuda

from .tools import image as tim
from .modeling import models


def call_model(images, call_model_args=None, expected_keys=None):
    """Generic function for getting predictions and gradients from a model."""
    target_class_idx =  call_model_args['class_id']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images.round())
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
                  call_model_args,
                  smooth=True,
                  methods='all',
                  xrai_mode='full',
                  batch_size=20):
    method_dict = {
        'gradient': 'GradientSaliency',
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
        if obj_name in ['GradientSaliency', 'GradCam', 'GuidedIG']:
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
                xrp.algorithm = xrai_mode
                obj_masks = [obj.GetMask(image,
                                         call_model,
                                         call_model_args,
                                         extra_parameters=xrp,
                                         batch_size=batch_size)]
                # All-white mask since XRAI doesn't do smoothing
                obj_masks += [np.zeros(obj_masks[0].shape) + 1.0]
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
               method,
               show=True,
               save=False,
               save_dir=None):
    # Set up the subplots
    fig, ax = plt.subplots(1, 5)
        
    # Fill hte panels
    tim.show_image(image / 255,
                   title='Original',
                   ax=ax[0])
    tim.show_image(masks[0],
                   cmap='gray',
                   title=method,
                   ax=ax[1])
    tim.show_image(masks[1],
                   cmap='gray',
                   title=method + ' (smooth)', 
                   ax=ax[2])
    tim.show_image(tim.overlay_heatmap(image, masks[0]),
                   title='Overlay',
                   ax=ax[3])
    tim.show_image(tim.overlay_heatmap(image, masks[1]),
                   title='Overlay (Smooth)',
                   ax=ax[4])
    plt.tight_layout()
    
    # Plot and save
    if save:
        assert save_dir, 'Need a directory to save the images.'
        plt.save(save_dir)
    if show:
        plt.show()
    
    return

