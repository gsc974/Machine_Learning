# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image     
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
from keras.preprocessing import image 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

def Count_label(diseaseID,Y_list):
    """
    Count target label
    Input Data : diseaseID, Y_list --> list OCT disease name, list OCT target
    Output Data : X_data --> Count number of images per disease
    """
    # Count files per disease
    X_data = [] # init X_data
    n = np.shape(np.unique(Y_list))[0] # Get number of unique data
    index = np.arange(n) # Array [0,1,2,3]
    for j in index:
        # Count True in Y_list==j
        X_data.append([np.sum(Y_list==j)])
    return X_data

def Plot_disease_count(Y1,Y2,Y3,Y_names,comments):
    """
    Plot Count target label
    Input Data : Y1,Y2,Y3,Y_names,comments --> Y_train,Y_test,Y_val,Y_names_train, comments
    """
    # Plot number of diseases per dataset
    plt.style.use('ggplot')
    # Create DataFrame for plot
    Y = np.array(np.concatenate((Count_label(Y_names,Y1), Count_label(Y_names,Y2),Count_label(Y_names,Y3)), axis=1))
    df = pd.DataFrame(Y, columns=['train','test','valid'], index=Y_names)
    # Show Dataframe
    # Plot bars
    ax=plt.figure(figsize=(13,7))
    for i,l in enumerate(df.columns):
        ax = plt.subplot(2,3,i+1)
        ax.set_title(comments[0] + l)
        bars = ax.bar(df.index,df[l],facecolor='cyan',edgecolor='black')
    plt.tight_layout()
    plt.show()


def Plot_image_check(X,X_path,datagen):
    """
    Plot raw images Vs preprocessed images
    Input Data : X,X_path,datagen --> Preprocessed images data, images files path, data generator for augmentation
    """
    # take subset of training data
    # X_subset = paths_to_tensor(X[:5]).astype('float32')
    X_subset = X[:5]
    #--------------------------
    # visualize RAW images
    fig = plt.figure(figsize=(20, 5))
    for i in range(0, len(X_subset)):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        img = image.load_img(X_path[i], target_size=(227, 227))
        ax.imshow(img)
    fig.suptitle('Subset of Raw Images', fontsize=20)
    plt.tight_layout()

    # visualize normailze images
    fig = plt.figure(figsize=(20, 5))
    for i in range(0, len(X_subset)):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(X_subset[i])
    fig.suptitle('Subset of Converted Images', fontsize=20)
    plt.tight_layout()

    # visualize augmented images
    fig = plt.figure(figsize=(20,5))
    for x_batch in datagen.flow(X_subset, batch_size=12):
        for j in range(0, len(X_subset)):
            ax = fig.add_subplot(1, 5, j + 1, xticks=[], yticks=[])
            ax.imshow(x_batch[j])
        fig.suptitle('Subset of Augemented Images', fontsize=20)
        plt.show()
        break
     

def plot_model_history(model_history):
    """
    Plot model history: Accuracy/Loss Vs Epoch
    Input Data : model_history --> history of model trained
    """ 
    # Plot model history 
    # Plot accuracy - Loss Vs epochs
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def Classification_ROC_Report(X,Y,model):
    """
    Plot Classification report | Confusion Matrix | ROC Curve
    Input Data : X,Y,model --> Preprocessed images, Target label, model
    """ 
    
    # Plot Classification report, Confustion Matrix, ROC
    labels = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

    # get predictions on the test set
    y_hat = model.predict(X)
    #
    Y_pred_classes = np.argmax(y_hat,axis = 1) 
    Y_true = np.argmax(Y,axis = 1)

    # Classification report
    ax=plt.figure(figsize=(15,5))
    ax = plt.subplot(1,3,1)
    rpt = sklearn.metrics.classification_report(np.argmax(Y, axis=1), np.argmax(y_hat, axis=1), target_names=list(labels.values()))
    ax.axis('off')
    ax.annotate(rpt, 
                 xy = (1.0,0.5), 
                 xytext = (0, 0), 
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=13, ha='right', va='center')  

    # Plot confusion matrix
    cm_df = Confusion_Matrix(Y,y_hat,labels,normalization=True)
    ax = plt.subplot(1,3,2)
    sns.heatmap(cm_df, annot=True)
    score = model.evaluate(X, Y, verbose=1)
    ax.set_title('Confusion Matrix\nresult: {0:.2f} - loss: {0:.2f}'.format(score[1], score[0]))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Plot ROC
    lw=2
    n_classes = 4
    fpr, tpr, roc_auc = ROC(Y,y_hat,n_classes)

    # Plot all ROC curves
    ax = plt.subplot(1,3,3)
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', '#4DBD33'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(labels[i], roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    return 

def Confusion_Matrix(Y,y_hat,labels,normalization=None):
    """
    Generate confusion matrix data frame
    Input Data : Y,y_hat,labels,normalization=None --> Groundthruth target label, Predicted target label, label list, plot data normalization
    Output Data : cm_df --> Confusion Matrix dataframe
    """ 
    #
    # Get prediction
    Y_pred_classes = np.argmax(y_hat,axis = 1) 
    Y_true = np.argmax(Y,axis = 1)

    # Creates a confusion matrix
    cm = confusion_matrix(Y_true, Y_pred_classes) 
    # Normalization parameter
    if normalization:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels.values(), 
                         columns = labels.values())

    return cm_df

def ROC(Y,y_hat,n_classes):
    """
    Generate ROC data
    Input Data : Y,y_hat,n_classes --> Groundthruth target label, Predicted target label, number of classes
    Output Data : fpr, tpr, roc_auc --> False Positive Rate, Tur Positive Rate, Area under curve
    """    
    # Compute ROC curve and ROC area for each class
    Y_pred_classes = y_hat
    Y_true = Y
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_pred_classes[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_pred_classes.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc

def Plot_predict(X,Y,model,X_path):
    """
    Plot prediction on validation dataset
    Input Data : X,Y,model,X_path --> Preprocessed images, Groundthruth target label, Model, image file path
    """       
    labels = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
    Y_pred_classes = np.argmax(model.predict(X),axis = 1) 
    Y_true = np.argmax(Y,axis = 1)
        
    fig = plt.figure(figsize=(40, 40))   
    for i in range(X.shape[0]):
        ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
        ax.set_title("Groundtruth : {} \n Prediction : {}".format(labels[Y_true[i]],labels[Y_pred_classes[i]]), \
                color=("green" if Y_true[i] == Y_pred_classes[i] else "red"),fontsize=20) 
        img = image.load_img(X_path[i])
        ax.imshow(img)
    plt.show()
    return
