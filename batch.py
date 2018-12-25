import os, cv2
import numpy as np
from keras.applications import imagenet_utils

class DataSet:
    '''
    to be filled out when I'm not procrastinating...
    '''
    def __init__(self, image_dir = 'dataset/train/images/', mask_dir='dataset/train/masks/'):
        #classes = self.classes
        image_path = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir)) if x.endswith('.png')]
        mask_path = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir)) if x.endswith('.png')]
        # transform list to array 
        image_path = np.array(image_path)
        mask_path = np.array(mask_path)
        
        # shuffle data by adding random permutation
        data_size = image_path.shape[0]
        random_index = np.random.permutation(data_size)
        image_path = image_path[random_index]
        mask_path = mask_path[random_index]
        
        # take eighty percent of the data as training 
        train_images = image_path[:int(data_size*0.8)]
        train_mask = mask_path[:int(data_size*0.8)]
        val_images = image_path[int(data_size*0.8):]
        val_masks = mask_path[int(data_size*0.8):]
        self.train_DataSet = ParseData(train_images, train_mask)
        self.validation_DataSet = ParseData(val_images, val_masks)
    def load_Dataset(self):
        return self.train_Dataset, self.validation_DataSet
    
    
    
class ParseData():
    '''
    testing testing....
    '''
    def __init__(self, image_path=None, mask_path=None):
        self.image_path = np.array(image_path)  
        self.mask_path = np.array(mask_path)
        self.batch_count = 0
        self.epoch_count = 0
        self.data_shape = 256*256
        self.classes = 5
    
    def next_batch(self,batch_size):
        start = self.batch_count * batch_size
        end = start + batch_size
        self.batch_count += 1       # update counter
        
        if end > self.image_path.shape[0]:
            self.batch_count = 0
            random_index = np.random.permutation(self.image_path.shape[0])
            self.image_path = self.image_path[random_index]
            self.mask_path = self.mask_path[random_index]
            self.epoch_count += 1
            start= self.batch_count * batch_size
            end = start * batch_size
            self.batch_count += 1
        image_batch, mask_batch = self.read_path(self.image_path[start:end], self.mask_path[start:end])
        return image_batch, mask_batch
    
    def read_path(self, image_path, mask_path):
        X = []
        Y = []
        
        for i in range(image_path.shape[0]):
            X.append(cv2.imread(image_path[i]))
            Y.append(self.onehot_it(cv2.imread(mask_path[i], cv2.CAP_MODE_GRAY)))
            
        return np.array(X), np.array(Y)
    
    def onehot_it(self, mask, classes = 5):
        mask = np.eye(classes)[mask]
        return mask
    def preprocess_inputs(self, X):
        return imagenet_utils.preprocess_input(X)
    
    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))
