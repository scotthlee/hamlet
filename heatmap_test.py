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
ims = [tim.load_image(p, (600, 600)) for p in im_paths]

# Loading a model
model = models.EfficientNet(num_classes=1, augmentation=False)
model.load_weights('output/binary/checkpoints/training/')
conv_layer = model.get_layer('top_conv')
model = tf.keras.models.Model([model.inputs], 
                              [conv_layer.output, model.output])
call_model_args = {'class_id': 0,
                   'model': model}

# Trying the single-image plot
methods = ['gradient', 'gradcam', 'xrai']
test_masks, test_methods = attribution.compute_masks(
                                image=im,
                                methods=methods,
                                call_model_args=call_model_args,
                                batch_size=1
)
pickle.dump(test_masks, open('img/method_masks.pkl', 'wb'))
attribution.panel_plot_by_method(image=im,
                                 masks=test_masks,
                                 methods=test_methods)

# Trying a multi-image plot, first with Grad-CAM, and then with XRAI
test_masks = [
    attribution.compute_masks(image,
                              methods=['gradcam'],
                              call_model_args=call_model_args,
                              batch_size=1)[0]
    for image in ims
]
test_masks = [m[0] for m in test_masks]
pickle.dump(test_masks, open('img/gradcam_masks.pkl', 'wb'))
attribution.panel_plot_by_image(images=ims[:3],
                                masks=test_masks[:3],
                                method_name='Grad-CAM',
                                save=True)

test_masks = [
    attribution.compute_masks(image,
                              methods=['xrai'],
                              xrai_mode='fast',
                              call_model_args=call_model_args,
                              batch_size=1)[0]
    for image in ims
]
test_masks = [m[0] for m in test_masks]
pickle.dump(test_masks, open('img/xrai_masks.pkl', 'wb'))
attribution.panel_plot_by_image(images=ims[:3],
                                masks=test_masks[:3],
                                method_name='xrai',
                                scale=2,
                                save=True)
