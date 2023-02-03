import numpy as np
import tensorflow as tf
import PIL.Image
import saliency.core as saliency
import numba
import gc
import os

from matplotlib import pyplot as plt
from numba import cuda
from copy import deepcopy

from . import models
from .tools import image as tim


def call_model(images, call_model_args=None, expected_keys=None):
    """Generic function for getting predictions and gradients from a model."""
    target_class_idx =  call_model_args['class_id']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images.round())
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            _, output_layer = model(images)
            output_layer = output_layer[:, target_class_idx]
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
            if obj_name == 'XRAI':
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

        gc.collect()

    return all_masks, obj_names


def panel_plot(images,
               masks,
               method_name='GradCam',
               smoothed=True,
               show=True,
               save=False,
               save_dir='img/',
               image_id=None,
               scale=1.5,
               xrai_cmap='gray',
               overlay_cmap='jet'):
    """Makes a single series of heatmaps for multiple images."""
    masks = deepcopy(masks)
    h = len(images)

    # Setting up the plots differently for XRAI and everything else
    if method_name in ['XRAI', 'xrai']:
        w = len(masks[0]) + 1
        fig, ax = plt.subplots(nrows=h,
                               ncols=w,
                               figsize=(scale * w, scale * h))
        titles = ['XRAI', 'XRAI (Top regions)']
        cmaps = [xrai_cmap] * 2
    else:
        w = len(masks[0]) + 2
        if smoothed:
            w += 1
            titles = ['Activations', 'Activations (Smooth)',
                      'Overlay', 'Overlay (Smooth)']
            cmaps = ['gray', 'gray', None, None]
        else:
            titles = ['Activations', 'Overlay']
            cmaps = ['gray', None]
        fig, ax = plt.subplots(nrows=h,
                               ncols=w,
                               figsize=(scale * w, scale * h))
        for i, image in enumerate(images):
            masks[i] += [tim.overlay_heatmap(image,
                                             masks[i][0],
                                             cmap=overlay_cmap)]
            if smoothed:
                masks[i] += [tim.overlay_heatmap(image,
                                                 masks[i][1],
                                                 cmap=overlay_cmap)]

    # Filling the plots
    if len(images) < 2:
        image = images[0]
        masks = masks[0]
        tim.show_image(image / 255, ax=ax[0])
        for j, mask in enumerate(masks):
            tim.show_image(mask,
                           cmap=cmaps[j],
                           ax=ax[j+1])

        titles = ['Original'] + titles
        for i, title in enumerate(titles):
            ax[i].set_title(title)

    else:
        for i, image in enumerate(images):
            tim.show_image(image / 255,
                           ax=ax[i, 0])
            for j, mask in enumerate(masks[i]):
                tim.show_image(mask,
                               cmap=cmaps[j],
                               ax=ax[i, j+1])

        titles = ['Original'] + titles
        for i, title in enumerate(titles):
            ax[0, i].set_title(title)

    # Adjusting space between and around the subplots
    if method_name != 'XRAI':
        plt.suptitle(method_name)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # Plotting and saving
    if image_id:
        save_path = save_dir + image_id + '_' + method_name + '.png'
    else:
        save_path = save_dir + method_name + '.png'
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()

    return
