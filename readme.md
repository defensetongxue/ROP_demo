# Demo for ROP-Marker: an evidence-oriented AI assistant for ROP diagnosis

[官中](./说明.md)

This repository is intended for reproducing or assisting in the annotation of Retinopathy of Prematurity (ROP) in preterm infants. Detailed development process (training and evaluation) codes can be found in the [main repository](https://github.com/defensetongxue/ROP_diagnoise). Currently, we are confirming relevant privacy protection regulations. At this stage, we only support providing trained models to relevant researchers. You can contact me via email at 1900013009@pku.edu.cn.

## Usage

After downloading and extracting the model in the root directory of the project, you will get a `modelCheckPoints` folder. This folder contains all the models trained in this project. You need to organize your files in the following order:

    -data_path 
    ---images
    -----1.jpg
    -----2.jpg
    -----...
    

`data_path` is the project directory where all intermediate results will be generated. All your image files should be in a folder named `images`.

Next, run `cleansing.py`. This script is designed to create an `annotations.json` file that retrieves paths of all files and stores them in a Python dictionary. It will also generate a `split` folder, where an `all.json` file will be created, using all images as test data by default. Note that our segmentation model currently processes images at the original dimensions captured by Retcam-3 (width 1600, height 1200). If the dimensions do not match, the performance may not be guaranteed. We attempted to evaluate our model on the publicly available HVDROP segmentation dataset and found that a small number of lesions were unrecognizable. We believe this is due to significant differences in image dimensions and cross-dataset shifts caused by different data styles. This experiment can be found in the `pretrain` branch at [ridge_segmentation](https://github.com/defensetongxue/ridge_segmentation). In this repository, you should ensure uniform image dimensions as much as possible. If difficulties arise, please resize to the aforementioned dimensions before performing any enhancements and further predictions.

Then, `imageEnhancer.py` will enhance all images. The enhanced images will be stored in `data_path/enhanced_image` and corresponding entries will be created in `annotations.json`.

`generate_mask.py` will analyze the pixels and generate a mask for the edges of each image. All masks will be stored in `data_path/mask` and corresponding entries will be created in `annotations.json`.

`optic_disc_location.py`, `ridge_segment.py`, `ROP_zone.py`, and `ROP_stage.py` will perform their respective operations. Note that `ROP_stage.py` depends on the intermediate results of `ridge_segment.py`, and `ROP_zone.py` depends on both `optic_disc_location.py` and `ridge_segment.py`. If you have any questions, feel free to let me know in the issues section.


