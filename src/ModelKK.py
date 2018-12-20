
# coding: utf-8

# In[13]:


import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
import tensorflow as tf


# In[22]:


def VoiceModel():
    
    #Construction of the LSTM model.
    
    #This is a many-to-many connected model.  
    model = Sequential()
    model.add(LSTM(256, input_shape=(20, 30), return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model


# In[23]:


#VModel = VoiceModel()

