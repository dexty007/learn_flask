#!/usr/bin/env python
# coding: utf-8

# ### Load libs

# In[45]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf



from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model

import os
import cv2


#get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)


# ### Helpers. Params. Preprocesing
# 

# In[46]:
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
        
        
def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)   


# In[47]:


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# In[48]:


#Test any image
def IsImageHasAnomaly(autoencoder, filePath,threshold):  
    im = cv2.resize(cv2.imread(filePath), (252, 252))
    im = im * 1./255
    datas = np.zeros((1,  252, 252, 3))
    validation_image[0, :, :, :] = im;   
    predicted_image = autoencoder.predict(validation_image)
    _mse = mse(predicted_image[0], validation_image[0]) 
    print('_mse: {}'.format(_mse))
    return _mse  > threshold


# In[49]:


img_width, img_height = 252, 252

batch_size = 252

nb_validation_samples=0
nb_train_samples=0

nb_epoch=50
model_dir = 'trained_models'

initial_image_dir='dataset/anamoly/data3'
train_data_dir = initial_image_dir + '/train'
validation_data_dir = initial_image_dir + '/valid'


# #### Generator for images to complete dataset
# Generator is used for extending the image dataset by image transformation

# In[50]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# #### New image generation flow

# In[71]:

'''
#image_list = os.listdir(initial_image_dir) #initial path to images
for root, dirs, files in os.walk(initial_image_dir):
    for img in files:
        print(root)
        inital_image_count=0
        #for img in image_list:   
        img_path= root + '/' + img
        if not os.path.isfile(img_path):
            continue

        inital_image_count += 1  

        img = load_img(img_path)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        train_save_to = train_data_dir + '/correct_samples'
        if not os.path.exists(train_save_to):
            os.makedirs(train_save_to)

        valid_save_to = validation_data_dir + '/correct_samples'
        if not os.path.exists(valid_save_to):
            os.makedirs(valid_save_to)

        print("generate additional images for train in: " + train_save_to)

        i = 0
        for batch in datagen.flow(x, batch_size=5, save_to_dir = train_save_to, save_prefix='sample', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

        train_size=0            
        for t in os.listdir(train_save_to):
            if os.path.isfile(train_save_to +"\\" + t):
                train_size += 1 


        print("generate additional images for validation in: " + valid_save_to)
        ii=0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=valid_save_to, save_prefix='doc', save_format='jpeg'):
            ii += 1
            if ii > 4:
                break  # otherwise the generator would loop indefinitely

        validation_size=0            
        for v in os.listdir(valid_save_to):
            if os.path.isfile(valid_save_to+"\\" +v):
                validation_size += 1 
            
print("-------------------------------------------")
print("Initial image count: {} ".format(inital_image_count))
print("Train image count: {} ".format(train_size))
print("Validation image count: {} ".format(validation_size))
'''

# ### Exract data for train and test

# In[51]:


# this is the augmentation configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),  
        batch_size=batch_size,
        color_mode='rgb', 
        class_mode=None)  

nb_train_samples=train_generator.samples
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb', 
        class_mode=None)

nb_validation_samples=validation_generator.samples


# ### Build Simplest Model

# In[52]:


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

autoencoder.summary()


# ### Train our network

# In[53]:
model_checkpoint = ModelCheckpoint(model_dir+'/model.{epoch:02d}.hdf5', 
                                        #monitor='val_acc', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        save_weights_only=False, 
                                        mode='auto', period=1)
callbacks = [model_checkpoint]

    

autoencoder.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=fixed_generator(validation_generator),
        validation_steps=nb_validation_samples // batch_size,
        use_multiprocessing=True,
        workers=32,
        callbacks=callbacks)


# ### Save weights

# In[54]:


autoencoder.save_weights('anomaly-detection.h5')


# ### Load weights

# In[55]:


autoencoder.load_weights('anomaly-detection.h5');


# ### Test encoder and visualize result

# In[56]:


img = next(validation_generator)[:7] # Get rendom image

dec = autoencoder.predict(img) # Decoded image
img = img[0]
dec = dec[0]
img = (img*255).astype('uint8')
dec = (dec*255).astype('uint8')

plt.imshow(np.hstack((img, dec)))
plt.title('Original and reconstructed images')
plt.show()


# ## Visual result

# Example of using mse

# In[57]:


#collect all mse-s
all_mses=[]
step=1;
for validation_image in validation_generator:   
    if step>nb_validation_samples:
        break;
        
    print(step, sep=' ', end='>', flush=True)       
    predicted_image = autoencoder.predict(validation_image)
    mse_value= mse(predicted_image[0], validation_image[0])
    all_mses.append(mse_value)
    step=step+1


# In[58]:


error_df = pd.DataFrame({'reconstruction_error':all_mses})
error_df.describe()


# In[59]:


fig = plt.figure()
ax = fig.add_subplot(111)

_ = ax.hist(error_df.reconstruction_error.values, bins=5)


#  Selecting th

# In[60]:


# base on visulization lets say that everething that more then 0.14 likelihood anomaly
# set threshold manually
threshold=0.14


# In[63]:


IsImageHasAnomaly(autoencoder, '10_left.jpeg',threshold)


# In[ ]:





