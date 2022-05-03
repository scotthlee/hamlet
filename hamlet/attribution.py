import numpy as np
import tensorflow as tf
import PIL.Image
import saliency.core as saliency
import numba
import gc
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


def xrai_percentile_mask(image, mask, level=70, scale=True):
    """Covers up parts of the image that don't meet the specified XRAI
    activation level.
    """
    out = np.array(image)
    meets_level = mask > np.percentile(mask, level)
    out[~meets_level] = 0
    if scale:
        out /= 255
    return out


def compute_masks(image,
                  call_model_args,
                  smooth=True,
                  methods=['gradient',
                           'gradcam', 
                           'xrai'],
                  xrai_mode='full',
                  xrai_level=70,
                  batch_size=1):
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
                obj_masks += [xrai_percentile_mask(image=image,
                                                   mask=obj_masks[0],
                                                   level=xrai_level)]
                all_masks += [obj_masks]
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
        
        if obj_name != 'XRAI':
            gray_masks = [saliency.VisualizeImageGrayscale(m) 
                          for m in obj_masks]
            all_masks += [gray_masks]
        
        # Free up some memory before next run; weird that this is needed, but
        # otherwise TF throws a memory allocation error the next time the
        # function is called.
        gc.collect()
    
    return all_masks, obj_names


def panel_plot(image, 
               masks, 
               methods,
               size=None,
               show=True,
               save=False,
               save_dir=None):
    # Set up the subplots
    fig, ax = plt.subplots(len(methods), 5, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for i, method in enumerate(methods):
        to_plot = [masks[i][0], masks[i][1]]
        # Set the panel titles and add empty subplots 
        if method == 'XRAI':
            shape = masks[i][0].shape
            titles = ['XRAI', 'XRAI (Top regions)', '', '']
            cmaps = ['gray'] * 4
            to_plot += [np.ones(shape)] * 2
        else:
            titles = [method, method +' (Smooth)', 'Overlay', 
                      'Overlay (Smooth)']
            cmaps = ['gray', 'gray', None, None]
            to_plot += [tim.overlay_heatmap(image, masks[i][0])]
            to_plot += [tim.overlay_heatmap(image, masks[i][1])]
        
        # Draw the original image
        if size:
            image = image.resize(size)
        tim.show_image(image / 255,
                       axis_off=False,
                       ax=ax[i, 0])
                
        # make xaxis invisibel
        ax[i, 0].set_ylabel(method)
        ax[i ,0].xaxis.set_visible(False)
        plt.setp(ax[i, 0].spines.values(), visible=False)
        ax[i, 0].tick_params(left=False, labelleft=False)
        
        # Fill in the sub panels
        for j, mask in enumerate(to_plot):
            tim.show_image(mask,
                           cmap=cmaps[j],
                           ax=ax[i, j+1])
    plt.tight_layout()
    if show:
        plt.show()
    
    return

