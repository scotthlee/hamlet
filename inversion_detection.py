'''Uses EfficientNet B0 to tell whether image colors are inverted.

Notes: 
  Training files should be in a subfolder named 'img/'. In our example, the 
    images for training the model were in 'inv/img'. The arg passed to 
    --train_image_dir should not include the 'img/' extension.
  If --model_folder is not specified, the model will be trained from scratch.
  If training a model from scratch, filenames for images with inverted
    colors should be prefixed with 'inv_'. 
'''

import numpy as np
import os
import argparse
import tensorflow as tf

from multiprocessing import Pool

from hamlet.modeling import models
from hamlet.tools.image import flip_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='D:/data/hamlet/',
                        help='directory holding the DICOM files')
    parser.add_argument('--train_img_dir',
                        type=str,
                        default='inv/',
                        help='subfolder in base_dir holding the folder \
                        that holds the training data')
    parser.add_argument('--check_img_dir',
                        type=str,
                        default='abn_train/',
                        help='subfolder in base_dir holding the folder \
                        that holds the images to be checked')
    parser.add_argument('--model_folder',
                        type=str,
                        default=None,
                        help='subfolder in checkpoints holding the model file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='minibatch size for the model')
    args = parser.parse_args()
    
    # Setting the directories
    BASE_DIR = args.base_dir
    TRAIN_DIR = args.base_dir + args.train_img_dir
    IMG_DIR = args.base_dir + args.check_img_dir
    LOG_DIR = 'output/inversion/logs/'
    CHECK_DIR = 'output/inversion/checkpoints/'
    
    # Parameters for the data loader
    BATCH_SIZE = 64
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    # Initializing a fresh model
    mod = models.EFficientNet(num_classes=1,
                              full_model=False,
                              img_height=IMG_HEIGHT,
                              img_width=IMG_WIDTH,
                              augmentation=False,
                              learning_rate=1e-2)
    
    if not args.model_folder:
        # Loading the files and generating labels
        img_files = os.listdir(TRAIN_DIR + 'img/')
        labels = np.array(['inv_' in f for f in img_files], dtype=np.uint8)
        
        # Making the training dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          TRAIN_DIR,
          labels=[l for l in labels],
          label_mode='int',
          validation_split=0.3,
          subset='training',
          seed=2022,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )
        
        # Making the validation dataset
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          TRAIN_DIR,
          labels=[l for l in labels],
          label_mode='int',
          validation_split=0.3,
          subset='validation',
          seed=2022,
          image_size=(img_height, img_width),
          batch_size=batch_size
        )
        
        # Setting up callbacks and metrics
        tr_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=1,
                                             restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=CHECK_DIR + 'training/',
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR + 'training/')
        ]
        
        # Fine-tuning the top layer of the model
        mod.fit(train_ds,
                validation_data=val_ds,
                callbacks=tr_callbacks,
                epochs=20)
    else:
        mod.load_weights(CHECK_DIR + args.model_folder)
    
    # Getting the predictions for the main set of x-rays
    files_to_check = os.listdir(IMG_DIR)
    new_ds = tf.keras.preprocessing.image_dataset_from_directory(
        IMG_DIR,
        labels=None,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    new_preds = mod.predict(new_ds, verbose=1).flatten().round()
    to_flip = np.where(new_preds == 1)[0]
    
    # Flipping images the model thinks are inverted
    with Pool() as p:
        input = [(files_to_check[i], IMG_DIR) for i in to_flip]
        p.starmap(flip_image, input)
        p.close()
        p.join()

