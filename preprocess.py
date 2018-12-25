import cv2, os
import numpy as np
import random

size = 256 # 
def generate_train_data(nb_of_patches=80000, size=256, image_dir="data/train/images/",
                        mask_dir="data/train/masks/"):
    '''
    Parameters:
        nb_of_pathces: The number of patches to be created
        image_dir:     Image directory
        mask_dir:      Corresponding mask directory
        size:          Image size. must be a single integer and not a tupple
    '''
    # Load image and mask directory as list
    image_path = [os.path.join(image_dir, x) for x 
                  in sorted(os.listdir(image_dir)) if x.endswith('.png')]
    mask_path = [os.path.join(mask_dir, y) for y
                 in sorted(os.listdir(mask_dir)) if y.endswith('.png')]
    
    # Directory to store patches (directory should be created in advace)
    train_image_path = "dataset/train/images/"
    train_mask_path = "dataset/train/masks/"
    
    # Total number of pathces per image
    patch_per_image = nb_of_patches // len(image_path)
    # variable to help identify patches
    image_id = 1
    
    for i in range(len(image_path)):
        count = 0
        image = cv2.imread(image_path[i])
        mask = cv2.imread(mask_path[i], cv2.CAP_MODE_GRAY) # numpy will map mask to image with 3 bands
        height, width = image.shape[0], image.shape[1]
        
        while count < patch_per_image:
            random_width = random.randint(0, width - size - 1)   # random width
            random_height = random.randint(0, height - size - 1) # random height
        
            # generate patches uisng random height and width
            img_patch = image[random_height: random_height + size, 
                              random_width: random_width + size, :]
            msk_patch = mask[random_height: random_height + size, 
                             random_width: random_width + size]
        
            # augment the data
            image_patch, mask_patch = data_augment(img_patch, msk_patch)
        
            # store the patches into train and label path
            # cv2.imwrite((train_image_path + '%05d.png' % image_id), image_patch)
            # cv2.imwrite((train_mask_path + '%05d.png' % image_id), mask_patch)
        
            cv2.imwrite((train_image_path + '%05d.png' % image_id), img_patch)
            cv2.imwrite((train_mask_path + '%05d.png' % image_id), msk_patch)
        
            #cv2.imwrite(('%05d.png' % image_id), img_patch)
            #cv2.imwrite(('%05d.png' % image_id), msk_patch)
        
            count +=1
            image_id +=1
            
# functions for data augmentation
def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)

    xb = cv2.warpAffine(xb, M_rotate, (size, size))

    yb = cv2.warpAffine(yb, M_rotate, (size, size))

    return xb, yb

def random_gamma_transform(img, gamma_vari): # random gamma transform
    log_gamma_vari = np.log(gamma_vari)

    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(size)]#255*((x/255)^gamma

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8) #project table

    return cv2.LUT(img, gamma_table) #look up table

def blur(img):
    img = cv2.blur(img, (3, 3)) #mean value filter

    return img

def add_noise(img):
    for i in range(size): # range is size of image (can also use img)

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img

def data_augment(a, b):
    #if np.random.random() < 0.25:
    #    a, b = rotate(a, b, 90)
        
    #if np.random.random() < 0.25:
    #    a, b = rotate(a, b, 180)
        
    #if np.random.random() < 0.25:
    #    a, b = rotate(a, b, 270)
        
    #if np.random.random() < 0.25:
    #    a = cv2.flip(a, 1)
    #    b = cv2.flip(b, 1)
        
    #if np.random.random() < 0.25:
    #    a = random_gamma_transform(a, 1.0)
        
    if np.random.random() < 0.25:
        a = blur(a)
        
    #if np.random.random() < 0.25:
    #    a = cv2.bilateralFilter(a, 9, 75, 75)
        
    if np.random.random() < 0.25:
        a = cv2.GaussianBlur(a, (5,5), 1.5)
        
    if np.random.random() < 0.2:
        a = add_noise(a)
        
    return a, b
