#This is a pre-trained model that predicts the type of animal as an input image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

def process_image_for_input(image_path):
    image = image_utils.load_img(image_path,target_size=(224,224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    return image #Returning the image ready for prediction
    
def make_a_prediction(image):
    predictions = model.predict(image)
    predicted_animal = decode_predictions(predictions,top=1)
    print("Predicted animal:",predicted_animal[0][0][1],"with an accuracy of",predicted_animal[0][0][2])
    
if __name__ == "__main__":
    model = VGG16(weights='imagenet')
    img_path = input("Enter image path for prediction: ")
    pred_img = process_image_for_input(img_path)
    make_a_prediction(pred_img)