# HaMLET
This package holds the code we used for the HaMLET project. There are 3 main sub-packages--`attribution`, `modeling`, and `tools`--along with a number of top-level scripts that go from extracting images from DICOM files all the way through modeling and inference. The sub-packages have their own README files, but for more information on the scripts, please see below.

## Image preprocessing
For our project, all of the x-rays came in as DICOM files, and they were often a bit messy, e.g., with burned-in text on them, or or with large solid black or white borders. We used the following scripts to clean them up a bit and get them ready for modeling.

1. [image_extraction.py](image_extraction.py) scans a source directory for DICOM files, extracts the files' image arrays, trims any solid borders from the images, and then saves them as `.png` files to an output directory. If your x-rays are in DICOM files, this is the place to start.
2. [text_removal.py](text_removal.py) uses [Tesseract](https://github.com/tesseract-ocr/tesseract) to find images with burned-in metadata and move them from their source directory to a new directory. The script could also be modified to obscure text in the images rather than removing them entirely by altering the code for the [check_text()](https://github.com/scotthlee/hamlet/blob/0b86d0b79636a1a5e8f22c6fc1d5d886e4b2cfd0/hamlet/tools/image.py#L124-L145) function in `tools.image`. 
3. [inversion_detection.py](inversion_detection.py) uses a lightweight model (EfficientNet B0) to tell whether the colors in images have been inverted, and, if so, invert them back to normal. This is mainly useful for x-rays that have already been exported from their original DICOM files (those have a `PhotometricInterpretation` parameter that specifies inversion).

## Modeling
1. [dataset_splitting.py](dataset_splitting.py) combines the images from our three main data sources and splits thems into training, validation, and test sets. It also selects which of the columns from the DS-3030 form to keep in the final structured dataset. This is less likely to be reusable for other projects, but we're including it here for the sake of transparency.
2. [binary.py](binary.py) and [multilabel.py](multilabel.py) train and test the binary and multilabel classification models, respectively.

Data loading for `binary.py` and `multilabel.py` is handled by [tf.data](https://www.tensorflow.org/guide/data) data generators. When specifying directories (e.g., `--img_dir`), the full path should be provided, unless otherwise noted, and the images themselves should be in subfolders named `img/` in those directories. Please see the command-line arguments in the scripts for more information.

## Inference
1. [generate_predictions.py](generate_predictions.py) runs inference for both model types on a new set of images in and writes an image-level CSV to the same directory with the predicted probabilities for each outcome.
2. [generate_heatmaps.py](generate_heatmaps.py) generates different kinds of saliency maps for a variety of images. Right now the only method supported is Grad-CAM, but we hope to add others soon.

## Requirements
The package was written in Python 3.8. For required dependencies, please see [requirements.txt](requirements.txt).
