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
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
#from skimage.measure import compare_ssim
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim

#get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)


# ### Helpers. Params. Preprocesing
# 

# In[46]:


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
    im = cv2.resize(cv2.imread(filePath), (420, 420))
    im = im * 1./255
    datas = np.zeros((1,  420, 420, 3))
    validation_image = np.zeros((1,  420, 420, 3),np.float32)
    validation_image[0, :, :, :] = im;   
    print(validation_image[0].shape)
    predicted_image = autoencoder.predict(validation_image)
    #predicted_image = predicted_image.astype(np.float32)
    #validation_image = cv2.cvtColor(validation_image[0], cv2.COLOR_BGR2GRAY)
    #predicted_image = cv2.cvtColor(predicted_image[0], cv2.COLOR_BGR2GRAY)
    #validation_image = tf.image.rgb_to_grayscale(validation_image, name=None)
    #predicted_image = tf.image.rgb_to_grayscale(predicted_image, name=None)
    #print(validation_image.shape)
    #mssim,score = compare_ssim(predicted_image, validation_image,full=True,multichannel=True) 
    _mse = mse(predicted_image[0], validation_image[0]) 
    print(_mse)
    #print(score)
    #print('_mse: {}'.format(score))
    return _mse  > threshold


# In[49]:


img_width, img_height = 420, 420

batch_size = 32

nb_validation_samples=0
nb_train_samples=0

nb_epoch=50

initial_image_dir='dataset/anamoly/data'
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



# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

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
    
    
    
    
    
    #predicted_image = predicted_image.astype(np.float32)
    #validation_image = validation_image.astype(np.float32)
    #predicted_image = cv2.cvtColor(predicted_image[0], cv2.COLOR_BGR2GRAY)
    #validation_image = cv2.cvtColor(validation_image[0], cv2.COLOR_BGR2GRAY)
    #validation_image = tf.image.rgb_to_grayscale(validation_image, name=None)
    #predicted_image = tf.image.rgb_to_grayscale(predicted_image, name=None)
    #print(validation_image.shape)
    #mssim,score = compare_ssim(predicted_image, validation_image,full=True,multichannel=True) 
    #all_mses.append(mssim)
    
    
    mse_value= mse(predicted_image[0], validation_image[0])
    all_mses.append(mse_value)
    step=step+1


# In[58]:


error_df = pd.DataFrame({'reconstruction_error':all_mses})
print(error_df.describe())


# In[59]:


fig = plt.figure()
ax = fig.add_subplot(111)

_ = ax.hist(error_df.reconstruction_error.values, bins=5)
fig.savefig('auotoencoder.png')

#  Selecting th

# In[60]:


# base on visulization lets say that everething that more then 0.14 likelihood anomaly
# set threshold manually
threshold=0.01


# In[63]:


print(IsImageHasAnomaly(autoencoder, 'Training loss.png',threshold))


# In[ ]:





