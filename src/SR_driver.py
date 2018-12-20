
# coding: utf-8

# In[6]:


import pyaudio
import wave
import keyboard
from SR_utils import *
from keras.models import load_model
import h5py
import numpy as np

CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


# In[ ]:


VModel = load_model('../model/VModel_final.h5')


# In[ ]:


WAVE_OUTPUT_FILENAME = "input/voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


frames = []

print("Press C to start recording then Ctrl+C to stop recording")
while True:
    if keyboard.is_pressed('c'):
        break
    else:
        pass
#input()
print("---Recording")
try:
    while(True):
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    pass

print("---Done Recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# In[7]:


utterance_length = 30
input_mfcc = extract_mfcc('../src/input/voice.wav', utterance_length)


# In[9]:


input_mfcc = np.expand_dims(input_mfcc, axis=0)
pred_digit = VModel.predict(input_mfcc)
print("Digit predicted: ", np.argmax(pred_digit))

print("Press any key to continue")
input()

