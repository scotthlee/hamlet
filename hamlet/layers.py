"""Custom Keras layers"""
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K


class Augmentation(layers.Layer):
    """Implements various image augmentations as a Layer."""
    def __init__(self,
                 flip=True,
                 contrast=(0.349, 1.346),
                 hue=0.127,
                 saturation=(0.382, 1.403),
                 brightness=0.525,
                 resize_factors=[0.7, 0.85, 1.15, 1.3],
                 **kwargs):
        """ Initializes the augmentation layer.
        
        Parameters
        ----------
        flip : bool, default=True
            Whether to apply random left-right flips during augmentation.
        hue : float, default=0.1
            Random factor for adjusting hue during augmentation.
        contrast : tuple of floats, default=(0.5, 2)
            Lower and upper bounds for the random contrast factor to be applied
            during augmentation.
        saturation : tuple of floats, default=(0.7, 1.25)
            Lower and upper bounds for the random saturation factor to be 
            applied during augmentation.
        brightness : float, default=0.2
            Max random factor for brightness adjustment during augmentation. 
            Must be greater than 0.
        """
        super(Augmentation, self).__init__(**kwargs)  
        self.flip = flip
        self.con = contrast
        self.hue = hue
        self.sat = saturation
        self.bright = brightness
        self.resize_factors = resize_factors   
    
    def call(self, input, training=None):
        if training is None:
            training = K.learning_phase()
        
        def augmented_image(image):
            image = tf.image.random_contrast(image, self.con[0], self.con[1])
            image = tf.image.random_hue(image, self.hue)
            image = tf.image.random_saturation(image, self.sat[0], self.sat[1])
            image = tf.image.random_brightness(image, self.bright) 
            if self.flip:
                image = tf.image.random_flip_left_right(image)
            if self.resize_factors:
                scale = np.random.choice(self.resize_factors)
                image_dims = image.get_shape().as_list()
                h, w = image_dims[1], image_dims[2]
                image = tf.image.resize_with_crop_or_pad(image,
                                                          int(h * scale),
                                                          int(w * scale))                                          
            return image
        
        if training == 1:
            output = augmented_image(input)
        else:
            output = input
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                       'flip': self.flip,
                       'contrast': self.con,
                       'hue': self.hue,
                       'saturation': self.sat,
                       'brightness': self.bright,
                       'resize': self.resize_factors
        })
        return config
