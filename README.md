# Harnessing Machine Learning to Eliminate Tuberculosis (HaMLET)

## Overview
This repository holds code for the HaMLET project, an internal CDC effort to use computer vision models to improve the quality of overseas health screenings for immigrants and refugees seeking entry to the U.S.. The project is the result of a collaboration between the Immigrant, Migrant, and Refugee Health Branch ([IMRHB](https://www.cdc.gov/ncezid/dgmq/focus-areas/irmh.html)) in the Division of Global Migration and Quarantine([DGMQ](https://www.cdc.gov/ncezid/dgmq/index.html)) and the Office of the Director in the Center for Surveillance, Epidemiology, and Laboratory Services ([CSELS](https://www.cdc.gov/csels/index.html)).

## Data
We had about 200,000 x-rays to work with for the entire project. For model training, we used about 110,000 x-rays, with labels coming from the primary reads by radiologists at the original screening sites. For validation and testing, we used about 16,000 x-rays, with labels coming from (often secondary) reads by radiologists from a small number of panel sites with large screening programs. All x-rays were collected as part of routine medical screenings for immigrants and refugees seeking entry to the U.S., which CDC helps to administer in collaboration with the Department of State (click [here](https://www.cdc.gov/immigrantrefugeehealth/about/medical-exam-faqs.html) to learn more about the screening program).

Our main goal was to pilot models for itnernal quality control, but as a sanity check, we also evaluated on our model existing openly-available chest x-ray datasets, including the Shenzhen and Montgomery County TB datasets, made available by the [National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/), and the [NIH chest x-ray dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). For the latter, we used additional expert labels for the test data provided by [Google](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) as part of their research efforts.

## Methods
### Tasks
The project centers on three main tasks:

1. Determining whether an x-ray is normal or abnormal (binary classification);
2. Determining whether an x-ray contains abnormalities suggestive of TB (multilabel classification); and
3. Localizing abnormalities in x-rays when they exist (object detection/image segmentation).

### Architecture
For feature extraction, we used [EfficientNetV2S](https://arxiv.org/abs/2104.00298) architecture, pretrained on ImageNet. For each model, we changed the dimensionality and loss function of the final dense layer to match its particular classification task, and we added an optional custom [image augmentation layer](https://github.com/scotthlee/hamlet/blob/99a2606fa43446f8dcd14e4408ede285d9ade088/hamlet/modeling/models.py#L44-L90) to make agumentation tunable with [KerasTuner](https://keras.io/keras_tuner/). The default augmentation values come from [this paper](https://pubs.rsna.org/doi/full/10.1148/radiol.2019191293) by folks at Google, Stanford, and the Palo Alto VA, though, and should work well without tuning. 

### Hardware
We trained our models on a scientific workstation with 24 logical cores, 128GB of ram, and a single NVIDIA TITAN X GPU (12GB memory). The relatively small amount of compute by today's standards limited the amount of experimentation and hyperparameter tuning we were able to do, and so we typically use default values for things when they're available (e.g., the learning rate for the optimizer, or the random image perturbations for augmentation).

## Code
Our code falls into two categories: core Python modules with functions and classes for extracting, preprocessing, and modeling images in DICOM files; and command-line scripts that use those modules to impelment various stages of our project. Both should be reusable for other projects, but just to keep things tidy, we put the core modules in their own [package](hamlet/), keeping the command-line scripts here. Read on for more information about the latter, and see the package [README](hamlet/README.md) for info about the former.

### Image preprocessing
For our project, all of the x-rays came in as DICOM files, and they were often a bit messy, e.g., with burned-in text on them, or or with large solid black or white borders. We used the following scripts to clean them up a bit and get them ready for modeling.

1. [image_extraction.py](image_extraction.py) scans a source directory for DICOM files, extracts the files' image arrays, trims any solid borders from the images, and then saves them as `.png` files to an output directory. If your x-rays are in DICOM files, this is the place to start.
2. [text_removal.py](text_removal.py) uses [Tesseract](https://github.com/tesseract-ocr/tesseract) to find images with burned-in metadata and move them from their source directory to a new directory. The script could also be modified to obscure text in the images rather than removing them entirely by altering the code for the [check_text()](https://github.com/scotthlee/hamlet/blob/0b86d0b79636a1a5e8f22c6fc1d5d886e4b2cfd0/hamlet/tools/image.py#L124-L145) function in `hamlet.tools.image`. 
3. [inversion_detection.py](inversion_detection.py) uses a lightweight model (EfficientNet B0) to tell whether the colors in images have been inverted, and, if so, invert them back to normal. This is mainly useful for x-rays that have already been exported from their original DICOM files (those have a `PhotometricInterpretation` parameter that specifies inversion).

### Modeling
1. [dataset_splitting.py](dataset_splitting.py) combines the images from our three main data sources and splits thems into training, validation, and test sets. It also selects which of the columns from the DS-3030 form to keep in the final structured dataset. This is less likely to be reusable for other projects, but we're including it here for the sake of transparency.
2. [binary.py](binary.py) and [multilabel.py](multilabel.py) train and test the binary and multilabel classification models, respectively.

Data loading for `binary.py` and `multilabel.py` is handled by [tf.data](https://www.tensorflow.org/guide/data) data generators. When specifying directories (e.g., `--img_dir`), the full path should be provided, unless otherwise noted, and the images themselves should be in subfolders named `img/` in those directories. Please see the command-line arguments in the scripts for more information.

### Inference
1. [generate_predictions.py](generate_predictions.py) runs inference for both model types on a new set of images in and writes an image-level CSV to the same directory with the predicted probabilities for each outcome.
2. [generate_heatmaps.py](generate_heatmaps.py) generates different kinds of saliency maps for a variety of images. Right now the only method supported is Grad-CAM, but we hope to add others soon.

### Requirements
The package was written in Python 3.8. For required dependencies, please see [requirements.txt](hamlet/requirements.txt).

### Visualization
We used the [saliency](https://github.com/PAIR-code/saliency) package from the [People+AI Research (PAIR)](https://github.com/PAIR-code) group at Google to make heatmaps that show where the models think there are abnormalities in the x-rays. Here's an exampe of some [Grad-CAM](https://arxiv.org/abs/1610.02391) heatmaps for our binary model's predictions on true abnormal x-rays. 

![grad cam](img/Grad-CAM_panel.png)

Grad-CAM often highlights parts of the image that wouldn't be important for making a dignosis--that is, it's not very specific as an abnormality localizer--but it does tend to capture the abnormalities when they're there. Trying another method, in this case [XRAI](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kapishnikov_XRAI_Better_Attributions_Through_Regions_ICCV_2019_paper.pdf), gives a different look at the same three images.

![XRAI](img/xrai_panel.png)

If you're using our code for your own project, try experimenting with other saliency algorithms, like [Integrated Gradients](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) and [Blurred Integrated Gradients](https://arxiv.org/pdf/2004.03383.pdf), to see which one works best. See the functions in the [attribution](hamlet/attribution.py) module for more info on how to run each method.

## Results
### Classification metrics
Coming soon.

### Embedding explorer
Coming soon.

## Related documents

* [Open Practices](doc/open_practices.md)
* [Rules of Behavior](doc/rules_of_behavior.md)
* [Thanks and Acknowledgements](doc/thanks.md)
* [Disclaimer](doc/DISCLAIMER.md)
* [Contribution Notice](doc/CONTRIBUTING.md)
* [Code of Conduct](doc/code-of-conduct.md)

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](doc/DISCLAIMER.md)
and [Code of Conduct](doc/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/privacy.html](http://www.cdc.gov/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page are subject to the [Presidential Records Act](http://www.archives.gov/about/laws/presidential-records.html)
and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).


**General disclaimer** This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  Github is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software.

