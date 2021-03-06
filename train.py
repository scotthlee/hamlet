"""Trains a model on one of the three classification tasks."""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import hamlet.tools.analysis as ta
from hamlet import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='abnormal',
                        help='Which prediction task to train on.',
                        choices=['abnormal', 'abnormal_tb', 'findings'])
    parser.add_argument('--training_type',
                        type=str,
                        default='base',
                        choices=['base', 'fine_tune', 'both'],
                        help='Whether to do base training, fine-tuning, or \
                        both. Only applies if --mode is either train or both')
    parser.add_argument('--data_dir',
                        type=str,
                        default='D:/data/hamlet/',
                        help='Path to the directory holding the three image \
                        dataset folders (train, val, and test) and the \
                        CSV file with the image-level labels and metadata.')
    parser.add_argument('--csv_name',
                        type=str,
                        default='samp.csv',
                        help='Name of the CSV file (including the file \
                        extension) holding the image-level labels and \
                        metadata.')
    parser.add_argument('--train_mod_folder',
                        type=str,
                        default=None,
                        help='Folder holding the model file to be used for \
                        training. Ignored if --mode is "test".')
    parser.add_argument('--model_flavor',
                        type=str,
                        default='EfficientNetV2S',
                        help='What pretrained model to use as the feature \
                        extractor.')
    parser.add_argument('--no_augmentation',
                        action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Minibatch size for model training and inference.')
    parser.add_argument('--progressive', 
                        action='store_true',
                        help='Whether to progressively unfreeze blocks during \
                        fine-tuning (for models that do not fit in GPU \
                        memory).')
    parser.add_argument('--train_all_blocks',
                        action='store_true',
                        help='Whether to unfreeze all blocks of the network \
                        at the beginning of training. Will bypyass fine \
                        tuning.')
    parser.add_argument('--starting_block',
                        type=int,
                        default=0,
                        help='For models that do not fit in memory: How many \
                        blocks to unfreeze at the start of fine-tuning.')
    parser.add_argument('--metric',
                        type=str,
                        default='val_loss',
                        help='Which metric to use for early stopping.')
    parser.add_argument('--metric_mode',
                        type=str,
                        default='min',
                        help='Whether to min or max the metric',
                        choices=['min', 'max'])
    parser.set_defaults(no_augmentation=False, 
                        progressive=False)
    args = parser.parse_args()
    
    # Parameters
    BASE_TRAIN = args.training_type in ['base', 'both']
    FINE_TUNE = args.training_type in ['fine_tune', 'both']
    AUGMENT = not args.no_augmentation
    MODEL_FLAVOR = args.model_flavor
    ALL_BLOCKS = not args.progressive
    BATCH_SIZE = args.batch_size
    STARTING_BLOCK = args.starting_block
    PROGRESSIVE = args.progressive
    TASK = args.task
    METRIC = args.metric
    METRIC_MODE = args.metric_mode
    
    # Directories
    BASE_DIR = args.data_dir
    OUT_DIR = 'output/' + args.task + '/'
    CHECK_DIR = OUT_DIR + 'checkpoints/'
    LOG_DIR = OUT_DIR + 'logs/'
    TRAIN_MOD_FOLDER = args.train_mod_folder
    
    # Just some info
    if AUGMENT:
        print('Augmentation on.')
    if ALL_BLOCKS:
        print('Training all blocks.\n')
    
    # Reading the labels
    records = pd.read_csv(BASE_DIR + args.csv_name, encoding='latin')
    if TASK == 'findings':
        LABEL_COL = [
            'infiltrate', 'reticular', 'cavity',
            'nodule', 'pleural_effusion', 'hilar_adenopathy',
            'linear_opacity', 'discrete_nodule', 'volume_loss',
            'pleural_reaction', 'other', 'miliary'
        ]
        NUM_CLASSES = len(LABEL_COL)
    else:
        LABEL_COL = TASK
        NUM_CLASSES = 1
    
    records[LABEL_COL] = records[LABEL_COL].fillna(0).astype(np.uint8)
    train = records[records.split == 'train'].reset_index(drop=True)
    val = records[records.split == 'val'].reset_index(drop=True)
    test = records[records.split == 'test'].reset_index(drop=True)
    dfs = [train, val, test]
    
    # Parameters for the data loader
    img_height = 600
    img_width = 600

    # Validation data loader--used for both training and testing
    val_dg = ImageDataGenerator()
    val_dir = BASE_DIR + 'val/img/'
    val_gen = val_dg.flow_from_dataframe(dataframe=val,
                                         directory=val_dir,
                                         x_col='file',
                                         y_col=LABEL_COL,
                                         class_mode='raw',
                                         shuffle=False,
                                         target_size=(img_height, 
                                                      img_width),
                                         batch_size=BATCH_SIZE)

    train_dg = ImageDataGenerator()
    train_dir = BASE_DIR + 'train/img/'
    train_gen = train_dg.flow_from_dataframe(dataframe=train,
                                             directory=train_dir,
                                             x_col='file',
                                             y_col=LABEL_COL,
                                             class_mode='raw',
                                             target_size=(img_height, 
                                                          img_width),
                                             batch_size=BATCH_SIZE)
    
    # Setting up a fresh model
    mod = models.EfficientNet(num_classes=NUM_CLASSES,
                              img_height=img_height,
                              img_width=img_width,
                              augmentation=AUGMENT,
                              learning_rate=1e-4,
                              model_flavor=MODEL_FLAVOR,
                              effnet_trainable=ALL_BLOCKS)
    
    # Setting up callbacks and metrics
    tr_callbacks = [
        callbacks.EarlyStopping(patience=1,
                                mode=METRIC_MODE,
                                monitor=METRIC,
                                restore_best_weights=True,
                                verbose=1),
        callbacks.ModelCheckpoint(filepath=CHECK_DIR + 'training/',
                                  save_weights_only=True,
                                  mode=METRIC_MODE,
                                  monitor=METRIC, 
                                  save_best_only=True),
        callbacks.TensorBoard(log_dir=LOG_DIR + 'training/')
    ]
    
    if TRAIN_MOD_FOLDER:
        mod.load_weights(CHECK_DIR + TRAIN_MOD_FOLDER)
    
    if BASE_TRAIN:
        mod.fit(train_gen,
                validation_data=val_gen,
                callbacks=tr_callbacks,
                epochs=20)

    if FINE_TUNE and not ALL_BLOCKS:
        # New callbacks for the fine-tuning phase
        ft_callbacks = [
            callbacks.EarlyStopping(patience=1,
                                    mode=METRIC_MODE,
                                    monitor=METRIC,
                                    restore_best_weights=True,
                                    verbose=1),
            callbacks.ModelCheckpoint(filepath=CHECK_DIR + 'fine_tuning/',
                                      save_weights_only=True,
                                      mode=METRIC_MODE,
                                      monitor=METRIC,
                                      save_best_only=True),
            callbacks.TensorBoard(log_dir=LOG_DIR + 'fine_tuning/')
        ]
        
        # Layer numbers for the block breaks
        b0_blocks = [20, 78, 121, 165, 194]
        b7_blocks = [64, 258, 406, 555, 659, 763]
        
        # Dropping the learning rate so things don't blow up
        K.set_value(mod.optimizer.learning_rate, 4e-4)
        
        # Progressively fine-tuning the blocks
        if not PROGRESSIVE:
            b7_blocks = [b7_blocks[STARTING_BLOCK]]
            STARTING_BLOCK = 0
        
        for b in b7_blocks[STARTING_BLOCK:]:
            for layer in mod.layers[-b:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
            
            mod.fit(train_gen,
                    validation_data=val_gen,
                    epochs=20,
                    callbacks=ft_callbacks)
