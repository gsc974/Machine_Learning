from keras_squeezenet import SqueezeNet
from keras.engine import InputLayer
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Conv2D, MaxPooling2D, Convolution2D, AveragePooling2D, Activation, Concatenate
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Flatten, Dense, merge
import keras
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.applications import VGG16
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
# from visual_callbacks import AccLossPlotter
# from keras.utils import multi_gpu_model

def model_training(model, callbacks, batch_size, epochs, datagen_train, datagen_test, X_train, Y_train, X_test, Y_test, Model_Name):
    history=model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=Y_train.shape[0] // batch_size,
                            epochs=epochs, 
                            verbose=1, 
                            callbacks=callbacks,
                            validation_data=datagen_test.flow(X_test, Y_test, batch_size=batch_size),
                            validation_steps = X_test.shape[0] // batch_size)
    return history

def HistoryTraining_Save(history, Model_Name):
    # Save epoch history in .json and csv format
    #
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = Model_Name + '.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    print('Save :' + hist_json_file)

    # or save to csv: 
    hist_csv_file = Model_Name + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    print('Save :' + hist_csv_file)

    return



