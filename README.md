# real-time-speech-digit-recognition
The project aims to predict the digit spoken by a user into a microphone in real time and uses Recurrent Neural Networks to process Sequential Data using `Python 3.6.3` `Keras` and `PyAudio 0.2.11`. 

# Description
This original model consists of LSTM(Long-Short Term Memory) layers along with Fully Connected layers with a Softmax Output.
The LSTM layers propagate data using a many-to-many architecture.

PyAudio is used for the real time microphone input from the user.

# Results
The model has an accuracy of 88% on the [Spoken Digit Dataset](https://www.kaggle.com/divyanshu99/spoken-digit-dataset) benchmark.

# Execution
To run the program, simply execute the `SR_driver.py` file in the src folder.
