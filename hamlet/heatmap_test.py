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

def panel_plot(image, masks, method):
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
    tim.show_heatmap(tim.overlay_heatmap(im, masks[0]),
               title='Overlay',
               ax=plt.subplot(ROWS, COLS, 4))
    tim.show_image(tim.overlay_heatmap(im, masks[1]),
                     title='Overlay (Smooth)',
                     ax=plt.subplot(ROWS, COLS, 5))
    plt.tight_layout()
    plt.show()


im_dir = '../img/test/'
im_files = os.listdir(im_dir)
im_paths = [im_dir + s for s in im_files]

m = models.EfficientNet(num_classes=1, augmentation=False)
m.load_weights('../output/binary/checkpoints/training/')
conv_layer = m.get_layer('top_conv')
model = tf.keras.models.Model([m.inputs], [conv_layer.output, m.output])

# Load the image
im = tim.load_image(im_paths[0], (600, 600))
call_model_args = {'class_id': 0, 'model': model}

# Construct the saliency object. This alone doesn't do anthing.
gradient_saliency = saliency.GradientSaliency()

# Compute the vanilla mask and the smoothed mask.
vanilla_mask_3d = gradient_saliency.GetMask(im, 
                                            call_model, 
                                            call_model_args)
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, 
                                                       call_model, 
                                                       call_model_args)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
masks = [vanilla_mask_grayscale, smoothgrad_mask_grayscale]

# Plotting
panel_plot(im, masks, 'Gradient')