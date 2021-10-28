#This script loads MNIST_CNN model in order to make predictions
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import tensorflow.keras as ks
import matplotlib
import numpy as np

def load_and_scale_image(image_path):
    image = ks.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    matplotlib.pyplot.imshow(image,cmap='gray')
    return image

def image_preprocessing(image):
    image = ks.preprocessing.image.img_to_array(image)
    image = image.reshape(1,28,28,1)
    image = image/255
    return image

if __name__ == "__main__":
    model = ks.models.load_model('MNIST_CNN')
    img_path = input("Enter image path: ")
    img = load_and_scale_image(img_path)
    newimg = image_preprocessing(img)
    prediction = model.predict(newimg)
    print("The number is",np.argmax(prediction))