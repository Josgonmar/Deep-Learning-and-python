import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import tensorflow.keras as ks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def import_and_create_model():
    base_model = ks.applications.VGG16(weights='imagenet',
        input_shape = (224,224,3),
        include_top = False)
    base_model.trainable = False
    inputs = ks.Input(shape=(224,224,3))
    x = base_model(inputs,training=False)
    x = ks.layers.GlobalAveragePooling2D()(x)
    outputs = ks.layers.Dense(6,activation='softmax')(x)
    model = ks.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    
def data_augmentation_and_training(model):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)
    train_dataset = datagen.flow_from_directory('fruits/train/', 
        target_size=(224,224), 
        color_mode='rgb', 
        class_mode="categorical")
    valid_dataset = datagen.flow_from_directory('fruits/valid/', 
        target_size=(224,224), 
        color_mode='rgb', 
       class_mode="categorical")
    model.fit(train_dataset,
          validation_data=valid_dataset,
          steps_per_epoch=train_dataset.samples/train_dataset.batch_size,
          validation_steps=valid_dataset.samples/valid_dataset.batch_size,
          epochs=10)
    return model
    
if __name__ == '__main__':
    mymodel = import_and_create_model()
    mymodel = data_augmentation_and_training(mymodel)
    
    mymodel.save('Fruit_CLSS')