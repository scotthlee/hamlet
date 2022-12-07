"""Writes a report with the models' predictions for a set of new images.

Notes:
  1. 'img_dir' points to the directory holding the folder that holds the image
    files for prediction. That folder should be named 'img/'.

"""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf

from hamlet import models
from hamlet.tools import metrics as tm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        help='Path to the directory holding the folder with \
                        the images for prediction.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Path where the script should dump the output \
                        files. Writes to img_dir by default.')
    parser.add_argument('--write_to',
                        type=str,
                        default=None,
                        help='Existing CSV file to which the predictions \
                        should be written. Must also ')
    parser.add_argument('--id_column',
                        type=str,
                        default='id',
                        help='Name for the column holding the image IDs.')
    parser.add_argument('--ab_mod_dir',
                        type=str,
                        default='output/abnormal/checkpoints/training/',
                        help='Path to the folder holding the trained \
                        abnormal/normal model.')
    parser.add_argument('--abtb_mod_dir',
                        type=str,
                        default='output/abnormal_tb/checkpoints/training/',
                        help='Path to the folder holding the trained \
                        abnormal-TB model.')
    parser.add_argument('--find_mod_dir',
                        type=str,
                        default='output/findings/checkpoints/training/',
                        help='Path to the folder holding the trained \
                        multilabel classification model.')
    parser.add_argument('--model_flavor',
                        type=str,
                        default='EfficientNetV2M',
                        help='What pretrained model to use as the feature \
                        extractor.')
    parser.add_argument('--no_augmentation',
                        action='store_true',
                        help='Specifies that the models should be built with \
                        the image augmentation layer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=12,
                        help='Minibatch size for inference.')
    parser.add_argument('--image_dim',
                        type=int,
                        default=600,
                        help='Either dimension of the image to be passed \
                        to the model.')
    parser.add_argument('--single_GPU',
                        action='store_true',
                        help='Turns off distributed (multi-GPU) training')
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Prefix for the predictions file.')
    parser.set_defaults(no_augmentation=False,
                        single_GPU=False)
    args = parser.parse_args()

    # Setting things up
    AUGMENT = not args.no_augmentation
    AB_MOD_DIR = args.ab_mod_dir
    ABTB_MOD_DIR = args.abtb_mod_dir
    FIND_MOD_DIR = args.find_mod_dir
    MODEL_FLAVOR = args.model_flavor
    IMG_DIR = args.image_dir
    IMG_DIM = args.image_dim
    BATCH_SIZE = args.batch_size
    OUT_DIR = IMG_DIR
    DISTRIBUTED = not args.single_GPU
    PREFIX = args.prefix
    WRITE_TO = args.write_to
    ID_COL = args.id_column
    if args.output_dir is not None:
        OUT_DIR = args.output_dir

    # Setting training strategy
    if DISTRIBUTED:
        print('Using multiple GPUs.\n')
        cdo = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cdo)
    else:
        strategy = tf.distribute.get_strategy()

    # Checking the existing CSV file to make sure it contains the specified
    # image ID column
    current_data = pd.read_csv(OUT_DIR + WRITE_TO)
    no_id = ID_COL + ' must be a valid column in the WRITE_TO CSV file.'
    assert ID_COL in current_data.columns.values, no_id

    # Setting the column labels for the multilabel task
    findings = [
        'infiltrate', 'reticular', 'cavity',
        'nodule', 'pleural_effusion', 'hilar_adenopathy',
        'linear_opacity', 'discrete_nodule', 'volume_loss',
        'pleural_reaction', 'other', 'miliary'
    ]

    # Loading the data
    test_files = os.listdir(IMG_DIR + 'img/')
    test_ids = [f[:-4] for f in test_files]
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      IMG_DIR,
      shuffle=False,
      image_size=(IMG_DIM, IMG_DIM),
      batch_size=BATCH_SIZE
    )

    preds_df = pd.DataFrame(test_ids, columns=[ID_COL])

    # Loading the trained model
    with strategy.scope():
        if FIND_MOD_DIR:
            multi_mod = models.EfficientNet(num_classes=len(findings),
                                            img_height=IMG_DIM,
                                            img_width=IMG_DIM,
                                            augmentation=AUGMENT,
                                            model_flavor=MODEL_FLAVOR)
            multi_mod.load_weights(FIND_MOD_DIR)
            multi_preds = multi_mod.predict(test_ds, verbose=1)
            preds_df[[f + '_prob' for f in findings]] = multi_preds

        if AB_MOD_DIR:
            ab_mod = models.EfficientNet(num_classes=1,
                                         img_height=IMG_DIM,
                                         img_width=IMG_DIM,
                                         augmentation=AUGMENT,
                                         model_flavor=MODEL_FLAVOR)
            ab_mod.load_weights(AB_MOD_DIR)
            ab_probs = ab_mod.predict(test_ds, verbose=1).flatten()
            preds_df['abnormal_prob'] = ab_probs

        if ABTB_MOD_DIR:
            abtb_mod = models.EfficientNet(num_classes=1,
                                           img_height=IMG_DIM,
                                           img_width=IMG_DIM,
                                           augmentation=AUGMENT,
                                           model_flavor=MODEL_FLAVOR)
            abtb_mod.load_weights(ABTB_MOD_DIR)
            abtb_probs = abtb_mod.predict(test_ds, verbose=1).flatten()
            preds_df['abnormal_tb_prob'] = abtb_probs

    if WRITE_TO:
        if '.png' in current_data[ID_COL][0]:
            current_data[ID_COL] = [s[:-4] for s in current_data[ID_COL]]
        current_data.sort_values(ID_COL, inplace=True)
        preds_df.sort_values(ID_COL, inplace=True)
        all_data = pd.merge(current_data, preds_df, on=ID_COL)
        all_data.to_csv(OUT_DIR + WRITE_TO, index=False)
    else:
        preds_df.to_csv(OUT_DIR + PREFIX + 'predictions.csv', index=False)
