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

from hamlet.modeling.models import EfficientNet
from hamlet.tools import analysis as ta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str,
                        help='Path to the directory holding the folder with \
                        the images for prediction.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Path where the script should dump the output \
                        files. Writes to img_dir by default.')
    parser.add_argument('--bin_mod_dir',
                        type=str,
                        default='output/binary/checkpoints/training/',
                        help='Path to the folder holding the trained \
                        binary classification model.')
    parser.add_argument('--multi_mod_dir',
                        type=str,
                        default='output/multilabel/checkpoints/fine_tuning/',
                        help='Path to the folder holding the trained \
                        multilabel classification model.')
    parser.add_argument('--bin_cut_csv',
                        type=str,
                        default='output/binary/stats/binary_stats.csv',
                        help='Path to a CSV file holding the decision \
                        threshold for the abnormal/normal prediction. The \
                        column with the threshold should be named "cutpoint".')
    parser.add_argument('--multi_cut_csv',
                        type=str,
                        default='output/multilabel/stats/multi_stats.csv',
                        help='Path to a CSV file holding the decision \
                        thresholds for each kind of abnormality. The column \
                        with the thresholds should be named "cutpoint", and \
                        the column with the findings should be named \
                        "finding".')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Specifies that the models should be built with \
                        the image augmentation layer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Minibatch size for inference.')
    parser.add_argument('--img_dim',
                        type=int,
                        default=600,
                        help='Either dimension of the image to be passed \
                        to the model.')
    parser.set_defaults(augment=False)
    args = parser.parse_args()
    
    # Globals
    AUGMENT = args.augment
    BIN_MOD_DIR = args.bin_mod_dir
    BIN_CUT_CSV = args.bin_cut_csv
    MULTI_MOD_DIR = args.multi_mod_dir
    MULTI_CUT_CSV = args.multi_cut_csv
    IMG_DIR = args.img_dir
    IMG_DIM = args.img_dim
    BATCH_SIZE = args.batch_size
    OUT_DIR = IMG_DIR
    if args.output_dir is not None:
        OUT_DIR = args.output_dir
    
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
    
    preds_df = pd.DataFrame(test_ids, columns=['id'])
    
    if BIN_MOD_DIR:
        bin_mod = EfficientNet(num_classes=1,
                               img_height=IMG_DIM,
                               img_width=IMG_DIM,
                               augmentation=AUGMENT)
        bin_mod.load_weights(BIN_MOD_DIR)
        bin_preds = bin_mod.predict(test_ds, verbose=1).flatten()
        preds_df['abnormal_prob'] = bin_preds
        if BIN_CUT_CSV:
            cuts = pd.read_csv(BIN_CUT_CSV)
            cut_types = cuts.cutpoint_type.values
            for t in cut_types:
                cut = cuts.cutpoint[cuts.cutpoint_type == t].values
                col_name = 'abnormal_' + str(t)
                preds_df[col_name] = ta.threshold(bin_preds, cut)
    
    if MULTI_MOD_DIR:    
        multi_mod = EfficientNet(num_classes=len(findings),
                                 img_height=IMG_DIM,
                                 img_width=IMG_DIM,
                                 augmentation=AUGMENT)
        multi_mod.load_weights(MULTI_MOD_DIR)
        multi_preds = multi_mod.predict(test_ds, verbose=1)
        preds_df[[f + '_prob' for f in findings]] = multi_preds
        multi_preds = pd.DataFrame(multi_preds, columns=findings)
        print(multi_preds)
        if MULTI_CUT_CSV:
            cuts = pd.read_csv(MULTI_CUT_CSV)
            cut_dict = dict(zip(cuts.finding, cuts.cutpoint))
            for f in findings:
                preds_df[f] = ta.threshold(multi_preds[f], cut_dict[f])
    
    preds_df.to_csv(OUT_DIR + 'predictions.csv', index=False)
