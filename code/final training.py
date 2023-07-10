#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import random
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
np.random.seed(1000)
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


# In[2]:


pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/Y_train.pickle","rb")
Y_train = pickle.load(pickle_in)

pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/X_valid.pickle","rb")
X_valid = pickle.load(pickle_in)

pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/Y_valid.pickle","rb")
Y_valid = pickle.load(pickle_in)

pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("F:/study/sem 10/cvpr/Hasib/index processed data/Y_test.pickle","rb")
Y_test = pickle.load(pickle_in)

print(f"X_train= {X_train.shape} Y_train= {Y_train.shape}")
print(f"X_valid= {X_valid.shape} Y_valid= {Y_valid.shape}")
print(f"X_test= {X_test.shape} Y_test= {Y_test.shape}")


# In[3]:


class_names = ['Density1Benign', 'Density1Malignant', 'Density2Benign', 'Density2Malignant', 'Density3Benign', 'Density3Malignant', 'Density4Benign', 'Density4Malignant']


# In[4]:


model = keras.Sequential([
    keras.Input(shape=(224,224,3)),
    layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(29, activation='softmax')
]) 
model.summary()


# In[5]:


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)


# In[6]:


h = model.fit(x=X_train, y=Y_train, epochs=10, validation_data=(X_valid, Y_valid), batch_size=64)


# In[8]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(h.history['accuracy'], '--', label='train accuracy')
plt.plot(h.history['val_accuracy'], '--', label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.plot(h.history['loss'], '--', label='train loss')
plt.plot(h.history['val_loss'], '--', label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(loc='upper right')

plt.show()


# In[ ]:




