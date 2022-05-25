""""Support classes and functions for Keras"""

import numpy as np
import pandas as pd
import itertools
import tensorflow as tf

from tensorflow.keras import optimizers, losses, layers, Model, Sequential
from tensorflow.keras import backend as K

from .layers import Augmentation


def EfficientNet(num_classes=1,
                multi_type='label',
                img_height=600,
                img_width=600,
                model_flavor='EfficientNetB7',
                n_channels=3,
                top_drop=0.5,
                augmentation=False,
                flip=True,
                hue=0.127,
                contrast=(0.349, 1.346),
                saturation=(0.382, 1.403),
                brightness=0.525,
                optimizer='adam',
                learning_rate=1e-2,
                effnet_trainable=False):
    """Builds EfficientNet with a custom top layer and optional augmentation.
    
    Parameters
    ----------
    num_classes : int, default=1
        The number of classes for prediction. 1 or 2 will train a binary
        model, and 3 or more will train a multilabel or multiclass model, 
        depending on the 'multi_type' parameter.
    multi_type : str, default='label'
        When num_classes > 2, whether to train the model as a multilabel or
        a multiclass model. Anything other than "label" will lead to 
        multiclass.
    img_width : int, default=600
        Width for the image after rescaling. Should match the dimensions 
        expected by the model specified by model_flavor.
    img_height : int, default=600
        Height for the image after rescaling. Should match the dimensions 
        expected by the model specified by model_flavor.
    model_flavor : kind of EfficientNet to run. Name must match the model 
        name in tf.keras.applications.efficientnet.
    n_channels : int, default=3
        Number of color channels in the image.
    top_drop : float, default=0.2
        Probabilility parameter for the top dropout layer.
    augmentation : bool, default=True
        Whether the first layer of the model should be an Augmentation layer.
    flip : bool, default=True
        Whether to apply random left-right flips during augmentation.
    hue : float, default=0.1
        Random factor for adjusting hue during augmentation.
    contrast : tuple of floats, default=(0.5, 2)
        Lower and upper bounds for the random contrast factor to be applied
        during augmentation.
    saturation : tuple of floats, default=(0.7, 1.25)
        Lower and upper bounds for the random saturation factor to be applied
        during augmentation.
    brightness : float, default=0.2
        Max random factor for brightness adjustment during augmentation. Must 
        be greater than 0.
    optimizer : str, default='adam',
        Optimizer to be used.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    
    Returns
    ----------
    A Keras model with an optional image aumgnetation layer, an EfficientNet
    feature extractor, and a top classification layer with untrained weights.
    """
    # Setting class-dependent hyperparameters
    loss = losses.BinaryCrossentropy()
    dense_activation = 'sigmoid'
    if num_classes != 1:
        err = '"multi_type" should be either "label" or "class"'
        assert multi_type in ['label', 'class'], err 
        if multi_type == 'class':
            loss = losses.CategoricalCrossentropy()
            dense_activation = 'softmax'
    
    # Piping the data into EFficientNet
    inputs = layers.Input(shape=(img_width, img_height, n_channels),
                          name='input')
    if augmentation:
        aug = Augmentation(flip=flip,
                           contrast=contrast,
                           hue=hue,
                           saturation=saturation,
                           brightness=brightness,
                           name='augmentation')(inputs)
    else:
        aug = inputs
    
    if 'V2' in model_flavor:
        effnet = getattr(tf.keras.applications.efficientnet_v2,
                         model_flavor)(include_top=False,
                                       input_tensor=aug,
                                       weights='imagenet')
    else:
        effnet = getattr(tf.keras.applications.efficientnet, 
                         model_flavor)(include_top=False,
                                       input_tensor=aug,
                                       weights='imagenet')
    effnet.trainable = effnet_trainable
    
    # Rebuilding the top layer for EfficientNet
    pool = layers.GlobalAveragePooling2D(name='avg_pool')(effnet.output)
    drop = layers.Dropout(top_drop, name='dropout')(pool)
    outputs = layers.Dense(num_classes, 
                           activation=dense_activation, 
                           name='dense')(drop)
    model = Model(inputs, outputs, name=model_flavor)
    
    # Adding metrics and compiling
    if (num_classes > 1) and (multi_type == 'label'):
        auc = tf.keras.metrics.AUC(name='ROC_AUC',
                                   num_labels=num_classes,
                                   multi_label=True)
    else:
        auc = tf.keras.metrics.AUC(name='ROC_AUC')
        
    metrics = [auc]
    
    # Setting the optimizer and compiling
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    
    return model
