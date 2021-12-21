#Deep learning neural network encoder using MNIST dataset.
#The output is expected to be the same as the input, with some quality loss.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from tensorflow.keras.datasets import mnist
import tensorflow.keras as ks
import matplotlib.pyplot as plot

def preprocess_data(x_tr,x_va):
    x_tr = x_tr.reshape(60000,28*28)
    x_va = x_va.reshape(10000,28*28)
    x_tr = x_tr/255
    x_va = x_va/255
    return(x_tr,x_va)

def create_and_compile_model():
    model = ks.Sequential()
    model.add(ks.layers.Dense(units=512,activation='relu',input_shape=(784,)))
    model.add(ks.layers.Dense(units=128,activation='relu'))
    model.add(ks.layers.Dense(units=10,activation='relu'))
    model.add(ks.layers.Dense(units=128,activation='relu'))
    model.add(ks.layers.Dense(units=512,activation='relu'))
    model.add(ks.layers.Dense(units=784,activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def predict_and_show(model):
    img_path = input("Enter image path: ")
    image = ks.preprocessing.image.load_img('test_images/' + img_path, color_mode="grayscale", target_size=(28,28))
    image = ks.preprocessing.image.img_to_array(image)
    image = image.reshape(1,784) / 255
    prediction = model.predict(image)
    plot.imshow(prediction.reshape(28,28))
    plot.gray()

if __name__ == "__main__":
    (x_train, _),(x_valid, _) = mnist.load_data()
    (x_train,x_valid) = preprocess_data(x_train, x_valid)
    autoencoder = ks.models.load_model('MNIST_AUTOENCODER')
    predict_and_show(autoencoder)
    
    #This part is only needed while training the model
    #autoencoder = create_and_compile_model()
    #autoencoder.fit(x_train,x_train,batch_size=256,epochs=50,verbose=1,validation_data=(x_valid,x_valid))
    #autoencoder.save('MNIST_AUTOENCODER')