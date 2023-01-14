# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 02:21:21 2022

@author: Anindya Chaudhuri
"""
# from sklearn.datasets import load_files       
# from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2 
from tqdm import tqdm

# COMMENT BELOW LINE - Udacityenvironment is outdated
#from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet50

# COMMENT BELOW LINE - Udacityenvironment is outdated
#from keras.preprocessing import image
import keras.utils as image                  

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D 
from keras.layers import Dense 


# load list of dog names
# for Udacity Environment - commented as testing locally
# dog_names = [item[20:-1] for item in sorted(glob("../../../data/dog_images/train/*/"))]

# for local environment
dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]


# load human image detector 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

'''
Write a Human Face Detector
'''
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

# COMMENT BELOW LINE - Udacityenvironment is outdated
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet import preprocess_input #, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']

'''
(IMPLEMENTATION) Model Architecture - Define your architecture.
'''
InceptionV3_model = Sequential()
InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
InceptionV3_model.add(Dense(133, activation='softmax'))

InceptionV3_model.summary()

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
##from extract_bottleneck_features import *

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def InceptionV3_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


### Algorithm 
def detect_image(img_path):
    if dog_detector(img_path):
        print("This photo contains a dog. It's {} breed.".format(InceptionV3_predict_breed(img_path).split('.')[1]))
        return "This photo contains a dog. It's {} breed.".format(InceptionV3_predict_breed(img_path).split('.')[1])
    
    #detect if it's a human
    if face_detector(img_path):
        print("This photo contains a human who looks like a(n) {}".format(InceptionV3_predict_breed(img_path).split('.')[1]))
        return "This photo contains a human who looks like a(n) {}".format(InceptionV3_predict_breed(img_path).split('.')[1])
    
    print("This photo contains neither a human nor a dog.")
    return "This photo contains neither a human nor a dog."


####################################
##### FLASK RELATED ################

from flask import Flask
from flask import render_template, request, flash,redirect,url_for
from werkzeug.utils import secure_filename
from glob import glob
import keras
import os

keras.backend.clear_session()
app = Flask(__name__)
app.secret_key= 'dog_breed_prediction'
app.config["IMG_FOLDER"] = 'static/uploads/' # 'images/'
app.config['MAX_CONTENT_LENGTH']= 16 * 1024 * 1024
ALLOWED_EXTENTIONS={'png','jpg','jpeg'}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENTIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

# web page that handles user query and displays model results
@app.route('/',methods=['POST'])
def upload_image():
    if request.method=='POST':

        if 'file' not in request.files:
            flash("Can't read file.")
            return redirect(request.url)

        file = request.files['file']

        if file.filename=='':
            flash("Image not selected, select an image.")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["IMG_FOLDER"],filename))
            flash("Image displayed successfully!!!")
            prediction = detect_image(app.config["IMG_FOLDER"]+filename)
            return render_template("master.html",filename=filename,prediction=prediction)
        else:
            flash('Image types of png, jpg and jpeg are allowed.')
            return redirect(request.url)
    else:
        redirect(request.url)

@app.route("/predict_display/<filename>")
def predict_display(filename):
    return redirect(url_for('static',filename='uploads/'+filename),code=301)

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
