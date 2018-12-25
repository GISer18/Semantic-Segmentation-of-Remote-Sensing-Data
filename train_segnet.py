#from keras.utils.training_utils import multi_gpu_model
import keras
import tensorflow as tf
import batch
from Models.model import SegNet
# parameters
input_shape = (256, 256, 3)
classes = 5
train_batch_size = 8000   # Number of images to be trained per batch
val_batch_size = 2000     # Number of image to validate on
batch_size = 128          # Mini batch size per each batch
epochs = 62               # Number of epochs per each batch

log_filepath ="./logs/"
optimizer = "adadelta"

# configure GPU use using tensorflow
config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True, per_process_gpu_memory_fraction=0.80))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

def main():
    # load the model 
    print("Loading model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    #model = multi_gpu_model(model, gpus = 4)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, 
                  metrics=["accuracy"])
    
    # load data
    print("Loading data")
    ds = batch.DataSet()
    parse = batch.ParseData()
    train_data = ds.train_DataSet
    val_data = ds.validation_DataSet
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1,
                                        write_graph=True, write_images=True)
    for steps in range(0, 8):
        X_train, Y_train = train_data.next_batch(train_batch_size)
        X_val, Y_val = val_data.next_batch(val_batch_size)
        print("Training data shape:", X_train.shape)
        #X_train = parse.preprocess_inputs(X_train)
        #X_val = parse.preprocess_inputs(X_val)
        
        Y_train = parse.reshape_labels(Y_train)
        Y_val = parse.reshape_labels(Y_val)
        print("Label data shape:", Y_val.shape)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1,
                  validation_data=(X_val, Y_val), shuffle=True, callbacks=[tb_cb])
        model.save('segnetpart.h5')
        
    model.save('segnet.h5')
    # Try using fit_generator() to load batches of data
    
if __name__ =="__main__":
    main()
