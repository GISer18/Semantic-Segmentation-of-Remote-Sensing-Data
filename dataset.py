import cv2, os
import numpy as np
from keras.applications import imagenet_utils


data_shape = 256*256

class Dataset:
    def __init__(self, classes=5, image_dir='dataset/train/images/', mask_dir='dataset/train/masks/'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.data_shape = 256*256
        self.classes = classes
    
    def data_generator(self,mode = 'train'):
        '''
        Load data for training. 
        TODO: add mode = 'testing' for validation data.
        '''
        image_dir = self.image_dir
        mask_dir = self.mask_dir
        #classes = self.classes
     
        # Create list of images
        image_path = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith('.png')]
        mask_path = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir)) if x.endswith('.png')]
        image_path = np.array(image_path)
        mask_path = np.array(mask_path)
        # Break data into testing and validation
        data_size = image_path.shape[0]
        random_index = np.random.permutation(data_size)
        image_path = image_path[random_index]
        mask_path = mask_path[random_index]
        
        # first 80 percent of dataset are for training
        X_tr = image_path[:int(data_size * 0.8)]
        Y_tr = mask_path[:int(data_size * 0.8)]
        # remaining subset is for validation
        X_val = image_path[int(data_size * 0.8):]
        Y_val = mask_path[int(data_size * 0.8):]
        
        # convert to np array
        if mode=='train':
            data, label = self.read_path(X_tr, Y_tr)
        elif mode=='test':
            data, label = self.read_path(X_val, Y_val)
        
        return data, label
            
    def read_path(self, image_path, mask_path):
        X = []
        Y = []
        for i in range(image_path.shape[0]):
            X.append(cv2.imread(image_path[i]))
            Y.append(self.onehot_label(cv2.imread(mask_path[i], cv2.CAP_MODE_GRAY)))
            
        return np.array(X), np.array(Y)
    
    def onehot_label(self, label):
        classes = self.classes
        mask = np.eye(classes)[label]
        return mask
    def preprocess_inputs(self, X):
        return imagenet_utils.preprocess_input(X)
    
    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))
    
    