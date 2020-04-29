from flask import Flask, url_for, send_from_directory, request,render_template
import logging, os, json
from werkzeug import secure_filename
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
import os 
import cv2
import shutil
import base64
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/test/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import tensorflow as tf
import keras
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_batchsize = 10   
image_size = 224      
test_dir = 'uploads'
    
def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/postjson', methods=['POST'])
def post():
    print(request.is_json)
    content = request.get_json()
    #print(content)
    print(content['id'])
    print(content['name'])
    return 'JSON posted'
    
@app.route('/upload', methods = ['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        #print(request.is_json)
        #content = request.get_json()
        print(request.headers)
        params = request.values
        names=params.get('name')
        ids=params.get('id')
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        isAnamoly,_mse , threshold = anamolyDetector(saved_path)
        if isAnamoly:
            base64img = ''
            with open('uploads/test/'+img_name,mode = 'rb') as image_file:
                    img = image_file.read()
                    base64img = base64.encodebytes(img).decode("utf-8")
            shutil.rmtree('uploads/test/')
            return '<html><title></title><body><h1><center>Uploaded Image is not valid Lungs X-Ray</center></h1><br/><h3><center>Threshold set: '+str(threshold)+' Error calculated:'  +str(_mse)+ '</center></h3><br/><center><img src="data:image/jpeg;base64,'+base64img+'"></<center></body></html>'
        predicted_classes = predict(img_name,names,ids,_mse , threshold )
        return predicted_classes
    else:
        return "<html><title></title><body><h1><center>Please upload an image</center></h1></body></html>"

import tensorflow as tf
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    keras.backend.set_session(session)
    new_model = load_model('model.h5')
    
def test_datagenerator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                              target_size=(image_size, image_size),
                              batch_size=test_batchsize,
                              class_mode='categorical',
                              shuffle=False) 
    return test_dir, test_generator

def predict(img_name,names,ids,_mse , threshold ):
    print('predicting on the test images...')
    test_dir1, test_generator = test_datagenerator()
    prediction_start = time.clock()            
    with session.graph.as_default():  
        keras.backend.set_session(session)    
        predictions = new_model.predict_generator(test_generator,
                                              steps=test_generator.samples / test_generator.batch_size,
                                              verbose=1)

    prediction_finish = time.clock()
    prediction_time = prediction_finish - prediction_start
    predicted_classes = np.argmax(predictions, axis=1)
    print(predictions)
    print(predicted_classes)
    filenames = test_generator.filenames
    data = {}
    for (a, b, c) in zip(filenames, predicted_classes, predictions): 
        data[a.split('\\')[len(a.split('\\'))-1]] = [a.split('\\')[len(a.split('\\'))-1],str(b),str(c)]  
    print("Predicted in {0:.3f} minutes!".format(prediction_time/60))
    base64img = ''
    with open('uploads/test/'+img_name,mode = 'rb') as image_file:
        img = image_file.read()
        base64img = base64.encodebytes(img).decode("utf-8")
    shutil.rmtree('uploads/test/')
    if b==0:
        d="covid"
    else:
        d="normal"
    listOfStr = ["covid", "normal"]
    zipbObj = zip(listOfStr, c)
    dictOfWords = dict(zipbObj)
    strhtml='<html><head><style>  body{background-color:white;}#customers {font-family: "Lucida Console", Monaco, monospace;border-collapse: collapse;border-radius: 2em;overflow: hidden;width:80%;height:45%;margin-top: 150px;margin-right: 150px;margin-left:150px;}#customers td, #customers th {border: 1px solid #ddd;padding: 8px;}#customers tr:nth-child(even){background-color: LavenderBlush;}#customers tr:hover {background-color: #ddd;}#customers th {padding-top: 12px;padding-bottom: 12px;text-align: left;background-color: DeepSkyBlue;color: white;}</style></head><body><h3><center>Threshold set: '+str(threshold)+' Error calculated:'  +str(_mse)+ '</center></h3><p></p><table id="customers" ><tr><th>PATIENT_NAME</th><th>PATIENT_ID</th><th>IMAGE_NAME</th><th>PREDICTED_CLASS</th><th>PREDICTIONS</th><th>IMAGE</th></tr><tr><td>'+names+'</td><td>'+ids+'</td><td>'+a[5:]+'</td><td>'+str(d)+'</td><td>'+str(dictOfWords)+'</td><td><img src="data:image/jpeg;base64,'+base64img+'"></td></tr></table></body></html>'
    return strhtml

def IsImageHasAnomaly(autoencoder, filePath,threshold):  
    im = cv2.resize(cv2.imread(filePath), (996, 996))
    im = im * 1./255
    datas = np.zeros((1,  996, 996, 3))
    validation_image = np.zeros((1,  996, 996, 3),np.float32)
    validation_image[0, :, :, :] = im;   
    predicted_image = autoencoder.predict(validation_image)
    _mse = mse(predicted_image[0], validation_image[0]) 
    app.logger.info('_mse: {}'.format(_mse))
    return _mse  > threshold , round(_mse, 4) , threshold
    
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
    
def anamolyDetector(inputImage):
    img_width, img_height = 996, 996
    input_img = Input(batch_shape=(None, img_width, img_width, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.load_weights('trained_models/anomaly-detection.h5');    
    threshold=0.020
    isAnamoly , _mse , threshold = IsImageHasAnomaly(autoencoder, inputImage,threshold)
    return isAnamoly , _mse , threshold

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001',debug=False)