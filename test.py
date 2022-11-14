"""Tests a model on one of the three classification tasks."""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from hamlet.tools import metrics as tm
from hamlet.tools import inference as ti
from hamlet import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--task',
                        type=str,
                        default='abnormal',
                        help='The target label for prediction.',
                        choices=['abnormal', 'abnormal_tb', 'findings'])
    parser.add_argument('--mod_folder',
                        type=str,
                        default='training/',
                        help='Folder holding the checkpoint for the trained \
                        model.')
    parser.add_argument('--model_flavor',
                        type=str,
                        default='EfficientNetV2M',
                        help='What pretrained model to use as the feature \
                        extractor.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=12,
                        help='Minibatch size for inference.')
    parser.add_argument('--no_augmentation',
                        action='store_true')
    parser.add_argument('--hamlet_only',
                        action='store_true')
    parser.add_argument('--other_only',
                        action='store_true')
    parser.add_argument('--single_GPU',
                        action='store_true',
                        help='Turns off distributed (multi-GPU) training')
    parser.add_argument('--keep_miliary',
                        action='store_true',
                        help='Keeps miliary findings as a potential target.')
    parser.set_defaults(no_augmentation=False,
                        hamlet_only=False,
                        other_only=False,
                        single_GPU=False,
                        keep_miliary=False)
    args = parser.parse_args()

    # Parameters
    AUGMENT = not args.no_augmentation
    MODEL_FLAVOR = args.model_flavor
    BATCH_SIZE = args.batch_size
    TASK = args.task
    OTHER_DATASETS = not args.hamlet_only
    OTHER_ONLY = args.other_only
    DISTRIBUTED = not args.single_GPU
    DROP_MILIARY = not args.keep_miliary

    # Directories
    DATA_DIR = args.data_dir
    HAM_DIR = DATA_DIR + 'hamlet/'
    OUT_DIR = 'output/' + TASK + '/'
    CHECK_DIR = OUT_DIR + 'checkpoints/'
    STATS_DIR = OUT_DIR + 'stats/'
    MOD_FOLDER = args.mod_folder

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
    val = records[records.split == 'val'].reset_index(drop=True)
    test = records[records.split == 'test'].reset_index(drop=True)
    dfs = [train, val, test]

    # Parameters for the data loader
    img_height = 600
    img_width = 600

    # Validation data loader--used for both training and testing
    val_dg = ImageDataGenerator()
    val_dir = HAM_DIR + 'val/img/'
    val_gen = val_dg.flow_from_dataframe(dataframe=val,
                                         directory=val_dir,
                                         x_col='file',
                                         y_col=LABEL_COL,
                                         class_mode='raw',
                                         shuffle=False,
                                         target_size=(img_height,
                                                      img_width),
                                         batch_size=BATCH_SIZE)

    test_dg = ImageDataGenerator()
    test_dir = HAM_DIR + 'test/img/'
    test_gen = test_dg.flow_from_dataframe(dataframe=test,
                                           directory=test_dir,
                                           x_col='file',
                                           y_col=LABEL_COL,
                                           class_mode='raw',
                                           shuffle=False,
                                           target_size=(img_height,
                                                        img_width),
                                           batch_size=BATCH_SIZE)

    # Loading the trained model
    with strategy.scope():
        mod = models.EfficientNet(num_classes=NUM_CLASSES,
                                  img_height=img_height,
                                  img_width=img_width,
                                  augmentation=AUGMENT,
                                  learning_rate=1e-4,
                                  model_flavor=MODEL_FLAVOR,
                                  effnet_trainable=True)

    mod.load_weights(CHECK_DIR + MOD_FOLDER)

    # Get the decision thresholds from the validation data
    if NUM_CLASSES == 1:
        if not OTHER_ONLY:
            val_probs = mod.predict(val_gen, verbose=1).flatten()
            cuts = ti.get_cutpoint(val[LABEL_COL], val_probs)

            # Writing validation the predictions
            val['abnormal_prob'] = val_probs
            val.to_csv(STATS_DIR + 'val_probs.csv')

            # Writing the test predictions
            test_probs = mod.predict(test_gen, verbose=1).flatten()
            test['abnormal_prob'] = test_probs
            test['abnormal_j'] = tm.threshold(test_probs, cuts['j'])
            test['abnormal_count'] = tm.threshold(test_probs, cuts['count'])
            test['abnormal_adj'] = tm.threshold(test_probs, cuts['count_adj'])
            test.to_csv(STATS_DIR + 'test_probs.csv')

            # And getting the test statistics
            stats = [tm.clf_metrics(test[LABEL_COL], test_probs, cuts[s])
                     for s in ['j', 'count', 'count_adj']]
            stats = pd.concat(stats, axis=0)
            stats['cutpoint'] = list(cuts.values())
            stats['cutpoint_type'] = pd.Series(['j', 'count', 'count_adj'])
            stats.to_csv(STATS_DIR + 'binary_stats.csv', index=False)

        if OTHER_DATASETS:
            try:
                ham_stats = pd.read_csv(STATS_DIR + 'binary_stats.csv')
                cut = ham_stats['cutpoint'][0].values
            except:
                cut = 0.5

            print('\nTesting on the NIH data.')
            nih_dg = ImageDataGenerator()
            nih_dir = DATA_DIR + 'nih/'
            nih_img_dir = nih_dir + 'test/img/'
            nih_df = pd.read_csv(nih_dir + 'test_labels.csv')
            nih_gen = nih_dg.flow_from_dataframe(dataframe=nih_df,
                                                 directory=nih_img_dir,
                                                 x_col='Image ID',
                                                 y_col='Abnormal',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 target_size=(img_height,
                                                              img_width),
                                                 batch_size=BATCH_SIZE)
            nih_preds = mod.predict(nih_gen, verbose=1).flatten()
            nih_stats = tm.clf_metrics(nih_df.Abnormal.values,
                                       nih_preds,
                                       cutpoint=cut)

            print('\nTesting on the Shenzhen data.')
            sh_dg = ImageDataGenerator()
            sh_dir = DATA_DIR + 'shenzhen/'
            sh_img_dir = sh_dir + 'images/img/'
            sh_df = pd.read_csv(sh_dir + 'shenzhen_metadata.csv')
            sh_gen = sh_dg.flow_from_dataframe(dataframe=sh_df,
                                               directory=sh_img_dir,
                                               x_col='study_id',
                                               y_col='abnormal',
                                               class_mode='raw',
                                               shuffle=False,
                                               target_size=(img_height,
                                                            img_width),
                                               batch_size=BATCH_SIZE)
            sh_preds = mod.predict(sh_gen, verbose=1).flatten()
            sh_stats = tm.clf_metrics(sh_df.abnormal.values,
                                      sh_preds,
                                      cutpoint=cut)

            print('\nTesting on the Montgomery County data.')
            mc_dg = ImageDataGenerator()
            mc_dir = DATA_DIR + 'mcu/'
            mc_img_dir = mc_dir + 'images/img/'
            mc_df = pd.read_csv(mc_dir + 'montgomery_metadata.csv')
            mc_gen = mc_dg.flow_from_dataframe(dataframe=mc_df,
                                               directory=mc_img_dir,
                                               x_col='study_id',
                                               y_col='abnormal',
                                               class_mode='raw',
                                               shuffle=False,
                                               target_size=(img_height,
                                                            img_width),
                                               batch_size=BATCH_SIZE)
            mc_preds = mod.predict(mc_gen, verbose=1).flatten()
            mc_stats = tm.clf_metrics(mc_df.abnormal.values,
                                      mc_preds,
                                      cutpoint=cut)

            other_stats = pd.concat([nih_stats, sh_stats, mc_stats],
                                    axis=0)
            other_stats['dataset'] = ['nih', 'shenzhen', 'montgomery']
            other_stats['model_type'] = [TASK] * 3
            other_stats.to_csv(STATS_DIR + 'other_stats.csv', index=False)

    else:
        # Check for existing prediction files
        pred_files = os.listdir(STATS_DIR)
        if 'val_preds.csv' not in pred_files:
            val_preds = pd.DataFrame(mod.predict(val_gen, verbose=1),
                                     columns=LABEL_COL)
            val_preds.to_csv(STATS_DIR + 'val_preds.csv', index=False)
        else:
            val_preds = pd.read_csv(STATS_DIR + 'val_preds.csv')

        if 'test_preds.csv' not in pred_files:
            test_preds = pd.DataFrame(mod.predict(test_gen, verbose=1),
                                      columns=LABEL_COL)
            test_preds.to_csv(STATS_DIR + 'test_preds.csv', index=False)
        else:
            test_preds = pd.read_csv(STATS_DIR + 'test_preds.csv')

        # Get the probabilities from the training dataset
        if DROP_MILIARY:
            LABEL_COL = LABEL_COL[:-1]

        p_vec = train[LABEL_COL].values.sum(0) / train.shape[0]
        cuts = ti.get_cutpoints(val[LABEL_COL], val_preds, p_adj=p_vec)
        stats = []
        for i, f in enumerate(LABEL_COL):
            j_stats = tm.clf_metrics(test[f],
                                      test_preds[f],
                                      cutpoint=cuts[f]['j'],
                                      p_adj=p_vec[i])
            count_stats = tm.clf_metrics(test[f],
                                          test_preds[f],
                                          cutpoint=cuts[f]['count_adj'],
                                          p_adj=p_vec[i])
            f_stats = pd.concat([j_stats, count_stats], axis=0)
            f_stats['finding'] = f
            f_stats['cutpoint'] = ['j', 'count_adj']
            stats.append(f_stats)

        stats = pd.concat(stats, axis=0)
        stats.to_csv(STATS_DIR + 'multi_stats.csv', index=False)

        prob_columns = [s + '_prob' for s in LABEL_COL]
        val_preds.columns = prob_columns
        test_preds.columns = prob_columns
        val = pd.concat([val, val_preds], axis=1)
        test = pd.concat([test, test_preds], axis=1)
        val.to_csv(STATS_DIR + 'val_preds.csv', index=False)
        test.to_csv(STATS_DIR + 'test_preds.csv', index=False)
