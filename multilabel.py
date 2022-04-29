"""Trains and tests a model to identify x-rays with specific abnormalities."""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tools import analysis as ta
from modeling import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        default='both',
                        choices=['train', 'test', 'both'],
                        help='Whether to train or test the model, or both.')
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
    parser.add_argument('--log_dir',
                        type=str,
                        default='../output/multilabel/logs/',
                        help='log_dir for Keras callbacks')
    parser.add_argument('--check_dir',
                        type=str,
                        default='../output/multilabel/checkpoints/',
                        help='check_dir for Keras callbacks')
    parser.add_argument('--stats_dir',
                        type=str,
                        default='../output/multilabel/stats/',
                        help='Where to save the test statstics.')
    parser.add_argument('--training_type',
                        type=str,
                        default='both',
                        choices=['base', 'fine_tune', 'both'],
                        help='Whether to do base training, fine-tuning, or \
                        both. Only applies if --mode is either train or both')
    parser.add_argument('--train_mod_folder',
                        type=str,
                        default=None,
                        help='Folder holding the model file to be used for \
                        training. Ignored if --mode is "test".')
    parser.add_argument('--test_mod_folder',
                        type=str,
                        default='fine_tuning/',
                        help='Folder holding the model file to be used for \
                        generating test predictions.')
    parser.add_argument('--augment',
                        action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Minibatch size for model training and inference.')
    parser.set_defaults(augment=False)
    args = parser.parse_args()
    
    # Parameters
    TRAIN = args.mode in ['train', 'both']
    TEST = args.mode in ['test', 'both']
    BASE_TRAIN = args.training_type in ['base', 'both']
    FINE_TUNE = args.training_type in ['fine_tune', 'both']
    AUGMENT = args.augment
    BATCH_SIZE = args.batch_size
    LOAD_WEIGHTS = True
    STARTING_BLOCK = 0
    
    # Directories
    BASE_DIR = args.data_dir
    LOG_DIR = args.log_dir
    CHECK_DIR = args.check_dir
    STATS_DIR = args.stats_dir
    TEST_MOD_FOLDER = args.test_mod_folder
    TRAIN_MOD_FOLDER = args.train_mod_folder
    
    # Reading the labels
    records = pd.read_csv(BASE_DIR + args.csv_name, encoding='latin')
    findings = [
        'infiltrate', 'reticular', 'cavity',
        'nodule', 'pleural_effusion', 'hilar_adenopathy',
        'linear_opacity', 'discrete_nodule', 'volume_loss',
        'pleural_reaction', 'other', 'miliary'
    ]
    records[findings] = records[findings].fillna(0).astype(np.uint8)
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
                                         y_col=findings,
                                         class_mode='raw',
                                         shuffle=False,
                                         target_size=(img_height, 
                                                      img_width),
                                         batch_size=BATCH_SIZE)

    # Training the top layer from scratch
    mod = models.EfficientNet(num_classes=len(findings),
                              multi_type='label',
                              img_height=img_height,
                              img_width=img_width,
                              augmentation=AUGMENT,
                              learning_rate=1e-3)

    if TRAIN:
        train_dg = ImageDataGenerator()
        train_dir = BASE_DIR + 'train/img/'
        train_gen = train_dg.flow_from_dataframe(dataframe=train,
                                                 directory=train_dir,
                                                 x_col='file',
                                                 y_col=findings,
                                                 class_mode='raw',
                                                 target_size=(img_height, 
                                                              img_width),
                                                 batch_size=BATCH_SIZE)

        # Setting up callbacks and metrics
        tr_callbacks = [
            callbacks.EarlyStopping(patience=1,
                                    restore_best_weights=True),
            callbacks.ModelCheckpoint(filepath=CHECK_DIR + 'training/',
                                      save_weights_only=True,
                                      monitor='val_loss',
                                      save_best_only=True),
            callbacks.TensorBoard(log_dir= + 'training/')
        ]
        
        if args.train_mod_folder:
            mod.load_weights(CHECK_DIR + args.train_mod_folder)

        if BASE_TRAIN:
            mod.fit(train_gen,
                    validation_data=val_gen,
                    callbacks=tr_callbacks,
                    epochs=20)

        if FINE_TUNE:
            # New callbacks for the fine-tuning phase
            ft_callbacks = [
                callbacks.EarlyStopping(patience=1,
                                        restore_best_weights=True),
                callbacks.ModelCheckpoint(filepath=CHECK_DIR + 'fine_tuning/',
                                          save_weights_only=True,
                                          monitor='val_loss',
                                          save_best_only=True),
                callbacks.TensorBoard(log_dir= + 'fine_tuning/')
            ]
            
            # Layer numbers for the block breaks
            b0_blocks = [20, 78, 121, 165, 194]
            b7_blocks = [64, 258, 406, 555, 659, 763]
            
            # Dropping the learning rate so things don't blow up
            K.set_value(mod.optimizer.learning_rate, 1e-4)
            
            # Progressively fine-tuning the blocks
            for b in b7_blocks[STARTING_BLOCK:]:
                for layer in mod.layers[-b:]:
                    if not isinstance(layer, layers.BatchNormalization):
                        layer.trainable = True
                
                mod.fit(train_gen,
                        validation_data=val_gen,
                        epochs=20,
                        callbacks=ft_callbacks)

    if TEST:
        test_dg = ImageDataGenerator()
        test_dir = BASE_DIR + 'test/img/'
        test_gen = test_dg.flow_from_dataframe(dataframe=test,
                                               directory=test_dir,
                                               x_col='file',
                                               y_col=findings,
                                               class_mode='raw',
                                               shuffle=False,
                                               target_size=(img_height, 
                                                            img_width),
                                               batch_size=BATCH_SIZE)

        # Getting the thresholds
        mod.load_weights(CHECK_DIR + TEST_MOD_FOLDER)
        val_preds = pd.DataFrame(mod.predict(val_gen, verbose=1),
                                 columns=findings)
        test_preds = pd.DataFrame(mod.predict(test_gen, verbose=1),
                                  columns=findings)
        cuts = ta.get_cutpoints(val[findings], val_preds)
        stats = [ta.clf_metrics(test[f], 
                                test_preds[f], 
                                cutpoint=cuts[f]['j'])
                 for f in findings]
        stats = pd.concat(stats, axis=0)
        stats['finding'] = findings
        stats['cutpoint'] = [cuts[f]['j'] for f in findings]
        stats.to_csv(STATS_DIR + 'multi_stats.csv', index=False)
