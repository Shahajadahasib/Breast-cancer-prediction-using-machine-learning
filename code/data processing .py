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


# In[2]:


DATA_DIR = 'F:\study\sem 10\cvpr\Hasib\index data'
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')


# In[3]:


gpu_device = tf.config.experimental.list_physical_devices('GPU')
print(f"Number of GPU = {len(gpu_device)}")
tf.config.experimental.set_memory_growth(gpu_device[0], True)


# In[4]:


CATEGORIES = []
IMG_SIZE = 224
for i in os.listdir(TRAIN_DATA_DIR):
    CATEGORIES.append(i)
    
print(CATEGORIES)


# In[5]:


training_data = []

for c in CATEGORIES:
    path = os.path.join(TRAIN_DATA_DIR, c) 
    class_num = CATEGORIES.index(c)
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img))  
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
            training_data.append([img_resized, class_num]) 
        except WException as e:
            pass
        
print(len(training_data))


# In[6]:


X_train = []
Y_train = []

for img, label in training_data:
    X_train.append(img)
    Y_train.append(label)

X_train = np.array(X_train).astype('float32')
Y_train = np.array(Y_train)

print(f"X_train= {X_train.shape} Y_train= {Y_train.shape}")


# In[7]:


print(X_train.shape)
print(Y_train.shape)
plt.imshow(X_train[2])
print(Y_train[1])


# In[8]:


test_data = []

for c in CATEGORIES:
    path = os.path.join(TEST_DATA_DIR, c) 
    class_num = CATEGORIES.index(c) 
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img))   
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
            test_data.append([img_resized, class_num])
        except WException as e:
            pass
        
print(len(test_data))


# In[9]:


random.shuffle(test_data)


# In[10]:


X_test = []
Y_test = []

for features,label in test_data:
    X_test.append(features)
    Y_test.append(label)

X_test = np.array(X_test).astype('float32')
Y_test = np.array(Y_test).astype('float32')

print(f"X_test= {X_test.shape} Y_test= {Y_test.shape}")


# In[11]:


plt.imshow(X_test[2])
print(Y_test[2])
type(Y_train[1])


# In[12]:


X_valid, Y_valid = X_train[:3000], Y_train[:3000]
random.shuffle(test_data)


# In[13]:


print(X_valid.shape)
print(Y_valid.shape)


# In[14]:


pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/X_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/Y_train.pickle","wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/X_valid.pickle","wb")
pickle.dump(X_valid, pickle_out)
pickle_out.close()

pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/Y_valid.pickle","wb")
pickle.dump(Y_valid, pickle_out)
pickle_out.close()

pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("F:\study\sem 10\cvpr\Hasib\index processed data/Y_test.pickle","wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()


# In[ ]:




