import os, cv2
import numpy as np
from augment import DataAugment

#
image_dir='/home/eric/Documents/PythonProjects/Ubinavi/data/train/images/'
mask_dir='/home/eric/Documents/PythonProjects/Ubinavi/data/train/masks/'

# image patch directory 
image_patch_dir = '/home/eric/Documents/PythonProjects/Ubinavi/dataset/train/images/'
masks_patch_dir = '/home/eric/Documents/PythonProjects/Ubinavi/dataset/train/masks/'

size = 512
augment = DataAugment() # class for data augmentation

def generate_patches(image_dir, mask_dir, size):
    """
    function to generate training set by creating patches of images from larger
    images, and then augmenting the data set by applying filters to image patches
    
    Input:
        size      -- height and width of image patch
        image_dir -- image directory
        mask_dir  -- image mask directory
    Returns: 
        a training set consisting of augmented image patches
    """
    # Create image and mask list from directory
    image_path = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith('.png')]
    mask_path = [os.path.join(mask_dir, y) for y in sorted(os.listdir(mask_dir)) if y.endswith('.png')]
    image_path = np.array(image_path)
    mask_path = np.array(mask_path)
    
    
    
    for image in range(len(image_path)):
        
        img = cv2.imread(image_path[image])
        msk = cv2.imread(mask_path[image], cv2.CAP_MODE_GRAY)
        height =  img.shape[0]
        width = img.shape[1]  # I'm fkn idiot I put a 0 here and spend a day wondering what's wrong
        step_h = height / size  # height step size
        step_w = width / size   # width step size
        img_id = 1
        for h in range(step_h):
            for w in range(step_w):
                # used to go through every image with same box size
                h_start = size * h 
                h_end =   size + size * h
                w_start = size * w
                w_end =   size + size * w
                
                img_patch = img[(h_start):(h_end), (w_start):(w_end), :]
                msk_patch  = msk[(h_start):(h_end), (w_start):(w_end)]
                
                # apply data augmentation
                
                # combination of median blurring and bilateral blur
                blur_img = augment.totalBlur(img_patch)
                blur_msk = augment.totalBlur(msk_patch)
                cv2.imwrite(image_patch_dir + str(img + 21) +'%03d.png' %img_id, blur_img)
                cv2.imwrite(masks_patch_dir + str(img + 21) +'%03d.png' %img_id, blur_msk)
                
                # speckle noise
                speckle_img = augment.speckle_noise(img_patch)
                speckle_msk = augment.speckle_noise(msk_patch)
                cv2.imwrite(image_patch_dir + str(img + 31) +'%03d.png' %img_id, speckle_img)
                cv2.imwrite(masks_patch_dir + str(img + 31) +'%03d.png' %img_id, speckle_msk)
                
                # elastic distortion
                distort_img = augment.distort_elastic_cv2(img_patch, alpha=60)
                distort_msk = augment.distort_elastic_cv2(msk_patch, alpha=60)
                cv2.imwrite(image_patch_dir + str(img + 41) +'%03d.png' %img_id, distort_img)
                cv2.imwrite(masks_patch_dir + str(img + 41) +'%03d.png' %img_id, distort_msk)
                
                # shift and rotation invariance
                rot_img = augment.rotation_invariance(img_patch)
                rot_msk = augment.rotation_invariance(msk_patch)
                cv2.imwrite(image_patch_dir + str(img + 51) +'%03d.png' %img_id, rot_img)
                cv2.imwrite(masks_patch_dir + str(img + 51) +'%03d.png' %img_id, rot_msk)
                
                
                cv2.imwrite(image_patch_dir + str(image+10) +'%03d.png' %img_id, img_patch)
                cv2.imwrite(masks_patch_dir + str(image+10) +'%03d.png' %img_id, msk_patch)
                
                img_id += 1
                
    
    return

def augment_patch(image_dir, mask_dir):
    """
    Augments patches of data using deformations
    from os.path import dirname, split, isdir
parent_dir = lambda x: split(x)[0] if isdir(x) else split(dirname(x))[0]
    """
    #from os.path import dirname, split, isdir
    # access parent directory
    #parent_dir = lambda x: split(x)[0] if isdir(x) else split(dirname(x))[0]
    
    image_list = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith(".png")]
    mask_list = [os.path.join(mask_dir, y) for y in sorted(os.listdir(mask_dir)) if y.endswith(".png")]
    
    img_array = np.array(image_list)
    msk_array = np.array(mask_list)
    
    img_id = 1
    for img in range(len(image_list)):
        img_patch = cv2.imread(img_array[img])
        msk_patch = cv2.imread(msk_array[img])
        
        # apply data augmentation
                
        # combination of median blurring and bilateral blur
        blur_img = augment.totalBlur(img_patch)
        blur_msk = augment.totalBlur(msk_patch)
        cv2.imwrite(image_patch_dir + str(img + 51) + '%03d.png' %img_id, blur_img)
        cv2.imwrite(masks_patch_dir + str(img + 51) + '%03d.png' %img_id, blur_msk)
        
        # speckle noise
        speckle_img = augment.speckle_noise(img_patch)
        speckle_msk = augment.speckle_noise(msk_patch)
        cv2.imwrite(image_patch_dir + str(img + 62) + '%03d.png' %img_id, speckle_img)
        cv2.imwrite(masks_patch_dir + str(img + 62) + '%03d.png' %img_id, speckle_msk)
        
        # elastic distortion
        distort_img = augment.distort_elastic_cv2(img_patch, alpha=60)
        distort_msk = augment.distort_elastic_cv2(msk_patch, alpha=60)
        cv2.imwrite(image_patch_dir + str(img + 73) + '%03d.png' %img_id, distort_img)
        cv2.imwrite(masks_patch_dir + str(img + 73) + '%03d.png' %img_id, distort_msk)
                
        # shift and rotation invariance
        rot_img = augment.rotation_invariance(img_patch)
        rot_msk = augment.rotation_invariance(msk_patch)
        cv2.imwrite(image_patch_dir + str(img + 84) + '%03d.png' %img_id, rot_img)
        cv2.imwrite(masks_patch_dir + str(img + 84) + '%03d.png' %img_id, rot_msk)
                
                

                
        img_id += 1
        
def main():
    print("generating patches...")
    
    generate_patches(image_dir=image_dir, mask_dir=mask_dir, size = size)
    
    
    
if __name__ == "__main__":
    main()
    
