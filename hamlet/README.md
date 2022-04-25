# HaMLET
This package holds the code we used for the HaMLET project. There are 4 main sub-packages--`preprocessing`, `modeling`, `attribution`, and `tools`--along with 4 top-level scripts that should be resuable for similar projects. Each sub-package has its own README with detailed information about how to run the modules as comand-line scripts, but for a brief overview of each, please see below.

## Image preprocessing
For our project, all of the x-rays came in as DICOM files, and they were often a bit messy, e.g., with burned-in text on them or large solid black or white borders. We used the preprocessing modules below to prep the images for modeling.

1. [image_extraction.py](preprocessing/image_extraction.py) scans a source directory for DICOM files, extracts the files' image arrays, trims any solid borders from the images, and then saves them as `.png` files to an output directory. 
2. [text_removal.py](preprocessing/text_removal.py) uses [Tesseract](https://github.com/tesseract-ocr/tesseract) to find images with burned-in metadata and move them from their source directory to a new directory. The script could also be modified to obscure text in the images rather than removing them entirely by altering the code for the [check_text()]() function in `tools.preprocessing`. 
3. [inversion_detection.py](preprocessing/inversion_detection.py) uses a lightweight model (EfficientNet B0) to tell whether the colors in images have been inverted, and, if so, invert them back to normal. This is mainly useful for x-rays that have already been exported from their original DICOM files (those have a `PhotometricInterpretation` parameter that specifies inversion).

The modules can be run in sequence as scripts to convert a batch of DICOM files to image files.

## Modeling
These modules contain the Keras layers and models we used for our project.
1. [models.py](modeling/models.py) contains the core Keras models used for training and inference. Right now the only model supported is EfficientNet, but we hope to add others soon
2. [layers.py](modeling/layers.py) contains the core custom Keras layers we used in the models.

## Attribtution
These modules support the generation of pixel-level saliency maps for x-rays. 
1. [grad-cam.py](attribution/grad_cam.py) uses Grad-CAM to generate heatmaps of model-predicted abnormalities. 
2. TBA

## Top-level scripts
1. [dataset_splitting.py](dataset_splitting.py) combines the images from our three main data sources and splits thems into training, validation, and test sets. It also selects which of the columns from the DS-3030 form to keep in the final structured dataset. This is less likely to be reusable for other projects, but we're including it here for the sake of transparency.
2. [binary.py](binary.py) and [multilabel.py](multilabel.py) train and test the binary and multilabel classification models, respectively.
3. [generate_predictions.py](generate_predictions.py) runs inference for both model types on a new set of images in and writes an image-level CSV to the same directory with the predicted probabilities for each outcome.
4. [generate_heatmaps.py](generate_heatmaps.py) generates different kinds of saliency maps for a variety of images. Right now the only method supported is Grad-CAM, but we hope to add others soon

Data loading for all scripts is handled by [tf.data](https://www.tensorflow.org/guide/data) data generators. When specifying directories (e.g., `--img_dir`), the full path should be provided, unless otherwise noted, and the images themselves should be in subfolders named `img/` in those directories. Please see the command-line arguments in the scripts for more information.

## Requirements
The package was written in Python 3.8. For required dependencies, please see [requirements.txt](requirements.txt).
