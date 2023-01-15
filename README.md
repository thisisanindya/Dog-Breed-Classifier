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
### We can do image Augmentation to improve performance further. 
### We can increase number of epoch and use early stopping to monitor metrics and stop training when performance on a validation dataset starts to decrease.

# Section 5: Web App Using Flask
## Prerequisite: 
### •	conda install -c conda-forge keras
### •	conda install tensorflow
### •	pip install opencv-python
## Also install waitress to better support flask
### •	conda install -c conda-forge waitress

## Steps to run the web app: 
### Run the web app from the command prompt: python runit.py
![image](https://user-images.githubusercontent.com/42642798/212484810-c1b612bc-4286-47cf-9f4f-1de2dcd36dfa.png)

### prompt shows flask app is running as displayed below: 
![image](https://user-images.githubusercontent.com/42642798/212485164-27cd6c46-3065-4e0f-98f5-7e7a37b61a5d.png)

### Open the browser and go to http://192.168.0.127:5000/. Below page opens up. 
![image](https://user-images.githubusercontent.com/42642798/212485181-3333c5c6-808c-443c-89a3-89a9b5210e8f.png)

### Click on “Choose File” and select “Brittany_02614.jpg” and click on open. 
![image](https://user-images.githubusercontent.com/42642798/212485200-7ae67f3c-19e3-4ae1-99c0-a8fd95344327.png)

### File name gets displayed on the screen. 
![image](https://user-images.githubusercontent.com/42642798/212485212-d8781c75-13b2-4916-9590-acf685288708.png)

### Click on submit. 
![image](https://user-images.githubusercontent.com/42642798/212485220-04000c30-5cc2-4969-b638-f168121fb058.png)

### Picture of the dog and the prediction displayed on the page.  Now select a different dog image say – “Brittany_02625.jpg”
![image](https://user-images.githubusercontent.com/42642798/212485261-5f71eeee-026e-437c-a18e-c15b9a109536.png)

### Picture of the dog and the prediction displayed on the page. 
![image](https://user-images.githubusercontent.com/42642798/212485808-d203d4a8-4bd2-4a3a-bcd1-10f140d7ee7c.png)

### Now select a different dog image say – “Golden_retriever_05186.jpg”. Click on Open and them submit. 
![image](https://user-images.githubusercontent.com/42642798/212485457-300142ab-54ef-4584-bed5-ecd6f12a3e64.png)

### Picture of the dog is and the prediction displayed on the page.
![image](https://user-images.githubusercontent.com/42642798/212485468-c732e9e8-40de-4f32-b0b0-29bcb4a6092b.png)

### Now select an image of a human, say “human-3.jpg” and then click open and submit. 
![image](https://user-images.githubusercontent.com/42642798/212485481-be03b83e-2bd0-49b5-a48a-14466ee7d3e8.png)

### The app identifies as a human face and predicts the corresponding dog breed. 
![image](https://user-images.githubusercontent.com/42642798/212485492-b9a343cb-9d75-4738-b3b9-57377ed1701c.png)

### Now finally select an image which  is neither of dog or of a human say - “circle.jpg”. Then click on open and submit. 
![image](https://user-images.githubusercontent.com/42642798/212485504-e393d27a-f496-4df5-bbba-451f236d8757.png)

### The app correctly suggests that - “This photo contains neither a human nor a dog." 
![image](https://user-images.githubusercontent.com/42642798/212485516-84b5c186-7bf6-4ca2-8976-343eaacd16c8.png)

### You can also refer to web-app.docx. 

# Section 6: Acknowledgement
### Dataset Credit - Thanks to Udacity for providing such a huge dataset. 
### Others: Thanks to Udacity for excellent course material and also to the mentors for timely help.  
