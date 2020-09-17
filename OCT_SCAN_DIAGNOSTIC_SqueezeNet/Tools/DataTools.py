from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing import image                  
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

def onehotcoding_string(Y):
    """
    One-Hot-Coding string
    Convert ['CNV','DME','DRUSEN','NORMAL'] --> [0,1,2,3]
    Input Data : Y --> array of string
    Output Data : Y --> array of integer
    """
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(Y)
    return Y

def onehotcoding(Y):
    """
    One-Hot-Coding 
    Convert [0,1,2,3] -->  [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    Input Data : Y --> array of integer
    Output Data : Y_Hot --> array of binary vector
    """
    Y_Hot = to_categorical(Y, num_classes = 4)
    return Y_Hot

def clean_dataset(X,Y):
    """
    Clean dataset
    Keep jpeg files only
    Input Data : X --> .jpeg files paths
    Output Data : Y --> OCT target class
    """
    X_Clean=[]
    Y_Clean=[]
    for i, x in enumerate(X):  # tqdm(enumerate(X))
        if (".jpeg") in x: 
            X_Clean.append(x)
            Y_Clean.append(Y[i]) 
    return np.array(X_Clean), np.array(Y_Clean)

def load_dataset(path):
    """
    Loading files path and keep .jpeg files only
    Input Data : path --> Folder 
    Output Data : data_file_list, data_target, data_target_names --> list of files path in folder and subfolder, 
    list of OCT target class, unique list of OCT target class  
    """
    # Load files path in 'data'
    print('Read Folder : ',path)
    data = load_files(path)
    # Save filenames and target
    data_file_list = np.array(data['filenames'])
    data_target = np.array(data['target'])
    data_target_names = np.array(data['target_names'])
    # Clean dataset
    data_file_list, data_target = clean_dataset(data_file_list, data_target)
    return data_file_list, data_target, data_target_names

def ReSamplingData(X,Y,ReSampleType):
    """
    Data resample
    Input Data : X,Y,ReSampleType --> list of .jpeg files, list of OCT target class associated, resampling Option
    resampling Option --> "ROS" : Random Over Sampling
                          "RUS" : Random Under Sampling   
    Output Data : X_resample, Y_resample --> list of .jpeg files resample, list of OCT target class associated resample
    """
    #
    # Reshape X 
    X_Flat = X.reshape(X.shape[0], 1)
    #
    if ReSampleType == 'ROS':
        resample = RandomOverSampler(ratio='auto',random_state=0)
    if ReSampleType == 'RUS':
        resample = RandomUnderSampler(ratio='auto',random_state=0)
    # Resample 
    X_resample, Y_resample = resample.fit_sample(X_Flat, Y)
    # Reshape X, Y
    X_resample = X_resample.reshape(X_resample.shape[0],)
    Y_resample = Y_resample.reshape(Y_resample.shape[0],) 
    #
    return X_resample, Y_resample

def path_to_tensor(img_path):
    """
    Convert input file data to tensorflow format
    Input Data : img_path --> .jpeg file path
    Output Data : x --> .jpeg file preprocessed  
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(227, 227))
    # convert PIL.Image.Image type to 3D tensor with shape (227, 227, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 227, 227, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert 
    x = preprocess_input(x)
    return x

def paths_to_tensor(img_paths):
    """
    Preprocess list of all .jpeg files
    Input Data : img_paths --> .jpeg files path
    Output Data : np.vstack(list_of_tensors) --> stack preprocessed images
    """
    # List files for loading and reshaping
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths] 
    return np.vstack(list_of_tensors)