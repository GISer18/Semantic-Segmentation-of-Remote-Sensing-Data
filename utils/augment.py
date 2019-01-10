import cv2
import numpy as np

class DataAugment:
    """
    class of functions used for data augmentation of image patches
    Uses pythons OpenCV binding
    """
        
    def gaussBlur(self, image, filter_size =15):
        '''
        gaussian blur
        '''
        blur = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
        
        return blur
    
    def medBlur(self, image, filter_size = 5):
        '''
        median blur
        '''
        blur = cv2.medianBlur(image, filter_size)
        
        return blur
    
    def bilateralBlur(self, image):
        '''
        bilateral filter
        '''
        blur = cv2.bilateralFilter(image, 9, 75, 75)
        
        return blur
    
    def totalBlur(self, image, filter_size=11):
        '''
        combination of median and bilateral filter
        '''
        mb = self.medBlur(image, filter_size)
        total = self.bilateralBlur(mb)
        
        return total
    
    
    def distort_elastic_cv2(self, image, alpha=180, sigma=5, random_state=None):
        '''
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
        '''
        
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        shape_size = image.shape[:2]
        
        # Downscale the random grid and then upsizing post filter
        # improve performance
        
        grid_scale = 4
        alpha //= grid_scale
        sigma //= grid_scale
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)
        
        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 -1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        
                
        rand_y = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
                
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])
        
        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)
        
        distorted_img = cv2.remap(image, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
        
        return distorted_img
    
    def rotation_invariance(self, img):
        """
        rotate and shift images randomly
        """
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        rows = img.shape[0]
        cols = img.shape[1]
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (rows, cols))
        return dst
    
    def speckle_noise(self, img, severity=10):
        '''
        add multiplicative speckle noise
        '''
        from skimage.util import random_noise
        
        speckle = random_noise(img, mode="speckle")
        for i in range(severity):
            speckle = random_noise(speckle, mode="speckle")
        return random_noise(speckle, mode="speckle")
    
    
def distort_elastic_cv2(image, alpha=150, sigma=8, random_state=None):
        
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        shape_size = image.shape[:2]
        
        # Downscale the random grid and then upsizing post filter
        # improve performance
        
        grid_scale = 4
        alpha //= grid_scale
        sigma //= grid_scale
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)
        
        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 -1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        
                
        rand_y = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
                
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])
        
        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)
        
        distorted_img = cv2.remap(image, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
        
        return distorted_img
        
    
