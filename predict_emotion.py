from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Define the emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

def load_and_predict(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Reshape the image to fit the model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.0
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    
    # Map the predicted class index to the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_class_index]
    
    return predicted_emotion

# Example usage
new_image_path = 'C:/Users/Fayaz S/OneDrive/Desktop/dl/ICCNN/1/3.jpg'  # Replace with the path to your new image
predicted_emotion = load_and_predict(new_image_path)
print(f'The predicted emotion is: {predicted_emotion}')