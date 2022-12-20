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
## Implementation: 
## Refinement: 

# Section 4: Results
## Model Evaluation and Validation: 
## Justification: 

# Section 5: Conclusion
## Reflection: 
## Improvement: 
