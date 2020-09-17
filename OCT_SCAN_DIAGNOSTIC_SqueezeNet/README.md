# Udacity Capstone Project 
# Machine Learning Nanodegree 2019

# Optical Coherence Tomography (OCT)

Retinal optical coherence tomography (OCT) is an imaging technique used to capture highresolution cross sections of the retinas of living patients.

The main goal of this capstone project is to classify automaticaly 3 types of ocular diseases and healthy ocular from OCT images by using ML technics. This capstone will pay attention on the evaluation of a Convolutionnal Neural Network (CNN) model called « SqueezeNet »

## Getting Started

### Prerequisites

Software environments used on AWS EC2@ p2.xlarge :
```
• Python 3.6.0 
• Tensorflow 1.12.0
• Tensorflow-gpu 1.13.0
• Keras 2.2.4
• cudnn 7.6.0
• cudatoolkit 10.0.130
```

### Data

Download OCT scan data from [Kaggle](https://www.kaggle.com/paultimothymooney/kermany2018)

## Repository content
```
• capstone_Report_OCT_SqueezeNet.pdf : Capstone report for OCT classification with SqueezeNet model
• OCT_SqueezeNet_ByPass_Model.ipynb : Jupyter notebook containing the best up to date results for OCT classification with SqueezeNet model.
• proposal.pdf : Capstone poposal for OCT classification with SqueezeNet model
• saved_model : saved_model folder contains .hdf5 files for model saved and .csv files for training log for different model testing
• Sensitivity_tests : Sensitivity_tests folder contains jupiter notebooks used for sensitivity studies, explained in capstone_Report_OCT_SqueezeNet.pdf
• Tools : Tools folder contains DataTools.py, ModelTools.py and PlotTools.py files. Each files contain function used for data pre/post processing 
```



