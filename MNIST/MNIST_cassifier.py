#Deep Learning neural network classifier using mnist dataset
from tensorflow.keras.datasets import mnist
import tensorflow.keras as ks

def reshape_prepare_data(x_tr,x_va):
    #Reshape images into a 28*28 pixels array
    x_tr = x_tr.reshape(60000,28*28)
    x_va = x_va.reshape(10000,28*28)
    #Normalize the data between 0 and 1
    x_tr = x_tr/255
    x_va = x_va/255
    return (x_tr,x_va)
    
def categorize_labels(y_tr,y_va):
    num_classes = 10
    y_tr = ks.utils.to_categorical(y_tr,num_classes)
    y_va = ks.utils.to_categorical(y_va,num_classes)
    return (y_tr,y_va)

def create_model():
    model = ks.Sequential()
    model.add(ks.layers.Dense(units=512,activation='relu',input_shape=(784,)))
    model.add(ks.layers.Dense(units=512,activation='relu'))
    model.add(ks.layers.Dense(units=10,activation='softmax'))
    return model

if __name__ == "__main__":
    (x_train,y_train),(x_valid,y_valid) = mnist.load_data()
    (x_train,x_valid) = reshape_prepare_data(x_train, x_valid)
    (y_train,y_valid) = categorize_labels(y_train, y_valid)
    
    mymodel = create_model()
    mymodel.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    mymodel.fit(x_train,y_train,epochs=5,verbose=1,validation_data=(x_valid,y_valid))
    
    mymodel.save('MNIST_SIMPLE')