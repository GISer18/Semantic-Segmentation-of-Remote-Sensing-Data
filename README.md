# Semantic-Segmentation-of-Remote-Sensing-Images
(Work In Progress)
This repository contains python code for dense semantic segmentation of remote sensing data using fully connected networks (FCN's). Thus far, it contains an implementation with Segnet. It will also include FCN ResNet, U-Net, and the pyramid scene parsing network (PSPNet). The purpose of this work is to study modern segmentation algorithms and their performance with regards to remote sensing data. 


**The Data Set**:

**1. Satellite Images**
Unfortunately the data set cannot be made publicly available for security reasons. The data I trained on is satellite image data which already had ground truth labels avalable. Because the images are large (~10,000 x 10,000 per image), I divided the images into small patches of size 256x256. This allowed me to have a dataset of 80,000 images,of which eighty percent was for training and the remaining twenty percent was used for validation.

There are several open source data sets which can be trained on. 
One such example is the kaggle challenge: [Dstl Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)

**2. Synthetic Aperture Radar Images**

The main goal of this project is to create an efficient FCN based algorithm to segment SAR images. To my knowledge there isn't an open source data set with labeled images. I did however find these images from [Jaxa](https://www.eorc.jaxa.jp/ALOS/en/guide/pal_10m_mosaic_dl.htm), which contain labeled data with 3 classes. 

There is also a lot of data available by the [Alaska Satellite facility](https://www.asf.alaska.edu/). You have to download their [mapready](https://www.asf.alaska.edu/data-tools/mapready/) software in roder to process the images, and then use the sentinel application platform[snap](http://step.esa.int/main/toolboxes/snap/) to create an image mask. 

**Preprocessing**

Because remote sensing data is large we need to break the images into smaller patches. The preprocessing.py script breaks the images into patches that can be trained on. The variable size in preprocessing can be adjusted to create patches as desired, while nb_of_patches is the number of pathces to be created. image_dir and mask_dir are the file directories

```python
size = 256 # patch size
def generate_train_data(nb_of_patches=80000, size=256, image_dir="data/train/images/",
                        mask_dir="data/train/masks/"):
  ```
 
Two folders need to be created to store the data. Create a folder named dataset, which contains a folder called train. Inside train will be two folder, images and masks, which will be used to store the image and label data. 
```python
# Directory to store patches (directory should be created in advace)
    train_image_path = "dataset/train/images/"
    train_mask_path = "dataset/train/masks/"
 ```
 
 ***WIP. To be continued** 
