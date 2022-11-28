"""Trains a model on one of the three classification tasks."""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from hamlet import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='abnormal',
                        help='Which prediction task to train on.',
                        choices=['abnormal', 'abnormal_tb', 'findings'])
    parser.add_argument('--data_dir',
                        type=str,
                        default='C:/Users/yle4/data/',
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
                        default='EfficientNetV2M',
                        help='What pretrained model to use as the feature \
                        extractor.')
    parser.add_argument('--no_augmentation',
                        action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=12,
                        help='Minibatch size for model training and inference.')
    parser.add_argument('--metric',
                        type=str,
                        default='val_ROC_AUC',
                        help='Which metric to use for early stopping.')
    parser.add_argument('--metric_mode',
                        type=str,
                        default='max',
                        help='Whether to min or max the metric',
                        choices=['min', 'max'])
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Turns on distributed (multi-GPU) training')
    parser.add_argument('--validate_on',
                        type=str,
                        default='hamlet',
                        choices=['hamlet', 'nih'],
                        help='Which dataset to use for validation.')
    parser.set_defaults(no_augmentation=False,
                        progressive=False,
                        distributed=False)
    args = parser.parse_args()

    # Parameters
    AUGMENT = not args.no_augmentation
    MODEL_FLAVOR = args.model_flavor
    BATCH_SIZE = args.batch_size
    TASK = args.task
    METRIC = args.metric
    METRIC_MODE = args.metric_mode
    DISTRIBUTED = args.distributed
    VALIDATE_ON = args.validate_on

    # Directories
    DATA_DIR = args.data_dir
    HAM_DIR = DATA_DIR + 'hamlet/'
    OUT_DIR = 'output/' + args.task + '/'
    CHECK_DIR = OUT_DIR + 'checkpoints/'
    LOG_DIR = OUT_DIR + 'logs/'
    TRAIN_MOD_FOLDER = args.train_mod_folder

    # Just some info
    if AUGMENT:
        print('Augmentation on.')

    # Setting training strategy
    if DISTRIBUTED:
        print('Using multiple GPUs.\n')
        cdo = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    else:
        strategy = tf.distribute.get_strategy()

    # Reading the labels
    records = pd.read_csv(HAM_DIR + args.csv_name, encoding='latin')
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

    # Parameters for the data loader
    img_height = 600
    img_width = 600

    # Loading the training data
    train_dg = ImageDataGenerator()
    train_dir = HAM_DIR + 'train/img/'
    train_gen = train_dg.flow_from_dataframe(dataframe=train,
                                             directory=train_dir,
                                             x_col='file',
                                             y_col=LABEL_COL,
                                             class_mode='raw',
                                             target_size=(img_height,
                                                          img_width),
                                             batch_size=BATCH_SIZE)

    # Loading the validation data
    val_dg = ImageDataGenerator()
    if VALIDATE_ON == 'hamlet':
        val = records[records.split == 'val'].reset_index(drop=True)
        val_dir = HAM_DIR + 'val/img/'
    elif VALIDATE_ON == 'nih':
        nih_labels = pd.read_csv(DATA_DIR + 'nih/labels.csv')
        val = nih_labels[nih_labels.split == 'val']
        val_dir = DATA_DIR + 'nih/val/img/'

    val_gen = val_dg.flow_from_dataframe(dataframe=val,
                                         directory=val_dir,
                                         x_col='file',
                                         y_col=LABEL_COL,
                                         class_mode='raw',
                                         shuffle=False,
                                         target_size=(img_height,
                                                      img_width),
                                         batch_size=BATCH_SIZE)

    # Setting up callbacks and metrics
    tr_callbacks = [
        callbacks.EarlyStopping(patience=2,
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

    with strategy.scope():
        # Setting up a fresh model
        mod = models.EfficientNet(num_classes=NUM_CLASSES,
                                  img_height=img_height,
                                  img_width=img_width,
                                  augmentation=AUGMENT,
                                  learning_rate=1e-4,
                                  model_flavor=MODEL_FLAVOR,
                                  effnet_trainable=True)

    if TRAIN_MOD_FOLDER:
        mod.load_weights(CHECK_DIR + TRAIN_MOD_FOLDER)

    mod.fit(train_gen,
            validation_data=val_gen,
            callbacks=tr_callbacks,
            epochs=20)
