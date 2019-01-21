from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

kernel = 3

def segnet(input_shape= (512, 512, 3), classes=5):
    
    img_input = Input(shape=input_shape)
    X = img_input
    
    # Encoding layers
    # Block 1 
    X = Convolution2D(64, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(64, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D()(X)
    
    # Block 2
    X = Convolution2D(128, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(128, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D()(X)
    
    # Block 3
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D()(X)
    
    # Block 4
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D()(X)
    
    # Block 5
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D()(X)
    
    # Decoding layers
    # Block 1
    X = UpSampling2D()(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    # Block 2
    X = UpSampling2D()(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(512, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    # Block 3
    X = UpSampling2D()(X)
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(256, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(128, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    # Block 4
    X = UpSampling2D()(X)
    X = Convolution2D(128, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(64, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    # Block 5
    X = UpSampling2D()(X)
    X = Convolution2D(64, (kernel, kernel), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Convolution2D(classes, (1, 1), padding="valid")(X)
    X = BatchNormalization()(X)
    X = Reshape((input_shape[0]*input_shape[1], classes), input_shape = input_shape)(X)
    X = Permute((1, 2))(X)
    X = Activation("softmax")(X)
    model = Model(img_input, X)
    return model
    