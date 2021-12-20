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
    image = image.reshape(1,784) #The input shape has to change depending on the
    image = image/255            #model you are using
    return image

if __name__ == "__main__":
    model = ks.models.load_model('MNIST_SIMPLE')
    img_path = input("Enter image path: ")
    img = load_and_scale_image('test_images/' + img_path)
    newimg = image_preprocessing(img)
    prediction = model.predict(newimg)
    print("The number is",np.argmax(prediction))