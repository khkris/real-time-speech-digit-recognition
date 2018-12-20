
# coding: utf-8

# In[3]:


import os
import librosa
import numpy as np


# In[4]:


# Obtain Mel-Frequency cepstral coefficients from the audio-input
def extract_mfcc(file_path, utterance_length):
    
    raw_audio, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain mfcc features from raw audio
    mfcc_features = librosa.feature.mfcc(raw_audio, sampling_rate)
    
    # Make sure mfcc_features is of utterance length
    if (mfcc_features.shape[1] > utterance_length):
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)
    
    return mfcc_features


# In[20]:


def get_mfcc_batch(file_path, batch_size, utterance_length):

    files = os.listdir(file_path)
    X_batch = []
    Y_batch = []

    while True:
        # Shuffle Files
        np.random.shuffle(files)
        for fname in files:

            # Make sure file is a .wav file
            if not fname.endswith(".wav"):
                continue
            
            # Get MFCC Features for the file
            mfcc_features = extract_mfcc(file_path + fname, utterance_length)
            
            # One-hot encode label for 10 digits 0-9
            y = np.eye(10)[int(fname[0])]
            
            # Append to label batch
            Y_batch.append(y)
            
            # Append mfcc features to ft_batch
            X_batch.append(mfcc_features)

            # Check to see if default batch size is < than ft_batch
            if len(X_batch) == batch_size:
                # send over batch
                yield X_batch, Y_batch
                # reset batches
                X_batch = []
                Y_batch = []

