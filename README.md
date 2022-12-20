# Dog-Breed-Classifier

# Section 1: Project Definition: 
## Project Overview: 
### In this project, we will build a model to process user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the breed. If supplied an image of a human, the code will identify the resembling dog breed.

## Problem Statement: 
### Our task is to build a an app that classifies the dog breeds if an image contains a dog or the resembling dog breed if an image of a human. This is a multi-class classification problem. The project uses Convolutional Neural Networks (CNNs). 

### This project consists of the following steps:
#### Exploratory Data Analysis
#### Detect any human faces in the image by OpenCV's implementation of Haar feature-based cascade classifiers.
#### Detect any dogs in the image by a pre-trained ResNet-50 model.
#### Classify dog breeds. W will try two CNN models to find the best ones:
#### Building CNN model from scratch
#### Building CNN model using transfer learning: VGG16
#### Building CNN model using transfer learning: InceptionV3
#### Use the training data to train model and validate the performance using the validation dataset according to metrics. Select the best performance model and save it.
#### Finally, we build a Flask app to detect dogs and humans from the input images by users.

## Metrics: 
### Accuracy metric used to measure the performance of a model as it is primarily used in a multi-class classification problems and works well for the well-balanced dataset as ours. 

# Section 2: Analysis
## Data Exploration, Visualization: 
### There are 133 total dog categories.
### There are 8351 total dog images.
### There are 6680 training dog images.
### There are 835 validation dog images.
### There are 836 test dog images.

# Section 3: Methodology
## Data Preprocessing: 
### Detect Humans - We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. 

### Detect Dogs - In this section, we use a pre-trained ResNet-50 model to detect dogs in images. 

## Implementation: 
### Detecting humans - We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors. In this project, we use haarcascades pre-trained face detectors stored as haarcascades/haarcascade_frontalface_alt.xml.

### Detecting Dogs - We use a pre-trained ResNet-50 model to detect dogs in images with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. 

### Classifying Dog Breeds - For classification, we try below CNN models to find the best one:
#### Building CNN model from scratch - CNN model from scratch without transfer learning. This has three convolutional layers followed by max-pooling layer. Next is the flatten layer to flatten the array to a vector. softmax activation function to convert the scores from the output into probability values.

#### Building CNN model using transfer learning: VGG16 - We use the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We add a global average pooling layer and a fully connected layer, where the latter has one node for each dog category and is supported by a softmax function.

#### Building CNN model using transfer learning: InceptionV3 - Finally we use InceptionV3 as a pre-trained CNN. It has a total of multiple layers and parts. This is more computationally efficient. Next is the GlobalAveragePooling2D layer. That takes 4D tensor (output of pre-trained CNN) as input and then returns 2D tensor by applying average pooling on the spatial dimensions. Finally, the Dense layer has a softmax activation function and 133 nodes, one node for each dog breed in the dataset.

## Refinement: For hyperparameter tuning, we use ModelCheckpoint of keras.callbacks with 25 epochs to measure accuracy and loss. The best modelinto a file.

# Section 4: Results - Model Evaluation, Validation, Justification
## The first model (CNN model from scratch) Test accuracy: 5.0239%. 
## The second model (CNN model with the pre-trained VGG-16 model) Test accuracy: 42.9426%
## The third model (CNN model with the pre-trained InceptionV3 model) Test accuracy: 79.3062%
### The second and third model were much better than the first one because of transfer learning. Also the third model is better than the second one because the InceptionV3 works better than VGG16 in the given dataset. 

# Section 5: Conclusion
## Reflection - We have classified dog breeds using CNNs. We detected human faces with Haar feature-based cascade classifiers and dogs with ResNet-50 transfer learning. Finally, we classifed dog breeds by CNN models with and without transfer learning VGG16 and Inception V3. 

## Improvement: 
### We can image Augmentation to improve furter performance. 
### we can increase number of epoch and use early stopping to monitor metrics and stop training when performance on a validation dataset starts to decrease.
