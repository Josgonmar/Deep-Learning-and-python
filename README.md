# Deep-Learning-with-python
Welcome to my repository. I'll be uploading all the work I'm working on, focused on Machine Learning with Tensorflow and Keras.

## MNIST:
This folder contains two deep learning networks, that are supposed to work with the dataset MNIST. It will tell you the digit its receiving as an input image of 28x28, from 0 to 9. The first one is a simple one, with just 3 DENSE layers, while the other one is using a convolutional neural network, a bit more complicated (even too complicated for the simple task it has to do), but I do this just for fun.
If you are in the mood, you can use the 'MNIST_pred.py' program to make a prediction with one of them.
Now there is also an Autoencoder using this dataset.
![alt text](https://github.com/Josgonmar/Deep-Learning-and-python/blob/main/Readme_files/Captura.PNG?raw=true)
## PRE-TRAINED:
In this folder there are some projects that make use of some pre-trainend Deep Learning models from the tensorflow.keras library for instance.
## TEXT GENERATOR
A deep learning model trained to predict a headline, using training data from thousands of New York Times headlines.
It is not very acurate, as not always the words hava a context or make sense inside the sentences, but it's fun to use.
Right now it predicts up to 5 more words added from the text seed you write as an input. Of course, it can be changed.
## LIBRARIES AND DEPENDENCIES:
- Python 3.9 (thats the version I'm using)
- Tensorflow 2
- Pandas
- Numpy
