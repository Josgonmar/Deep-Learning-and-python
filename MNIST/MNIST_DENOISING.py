#Deep learning neural network encoder using MNIST dataset.
#The output is expected to be the same as the input, with some quality loss.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from tensorflow.keras.datasets import mnist
import tensorflow.keras as ks
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data(x_tr,x_va):
    x_tr = x_tr.reshape(60000,28,28,1)
    x_va = x_va.reshape(10000,28,28,1)
    x_tr = x_tr/255
    x_va = x_va/255
    return(x_tr,x_va)

def noise_image(image):
    noise_factor = 0.5
    noisy = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    noisy_image = np.clip(noisy, 0., 1.)
    return noisy_image

def create_and_compile_model():
    model = ks.Sequential()
    model.add(ks.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(ks.layers.MaxPool2D((2,2), padding='same'))
    model.add(ks.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(ks.layers.MaxPool2D((2,2), padding='same'))
    model.add(ks.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(ks.layers.UpSampling2D((2,2)))
    model.add(ks.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(ks.layers.UpSampling2D((2,2)))
    model.add(ks.layers.Conv2D(1, (3,3), padding='same', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model

def predict_and_show(model):
    img_path = input("Enter image path: ")
    image = ks.preprocessing.image.load_img('test_images/' + img_path, color_mode="grayscale", target_size=(28,28))
    image = ks.preprocessing.image.img_to_array(image)
    image = image.reshape(1,28,28,1) / 255
    n_image = noise_image(image)
    prediction = model.predict(n_image)
    fig, ax = plt.subplots(2)
    ax[0].imshow(n_image.reshape(28,28))
    ax[0].set_title('Noisy input image')
    ax[0].axis('off')
    ax[1].imshow(prediction.reshape(28,28))
    ax[1].set_title('Denoised output image')
    ax[1].axis('off')

if __name__ == "__main__":
    (x_train, _),(x_valid, _) = mnist.load_data()
    (x_train,x_valid) = preprocess_data(x_train, x_valid)
    x_train_noisy = noise_image(x_train)
    x_valid_noisy = noise_image(x_valid)
    autoencoder = ks.models.load_model('MNIST_DENOISER')
    predict_and_show(autoencoder)
    
    #This part is only needed while training the model
    #denoiser = create_and_compile_model()
    #denoiser.fit(x_train_noisy,x_train,batch_size=256,epochs=50,verbose=1,validation_data=(x_valid_noisy,x_valid))
    #denoiser.save('MNIST_DENOISER')