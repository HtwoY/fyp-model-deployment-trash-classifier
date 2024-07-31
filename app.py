import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('tuned_efficientnetb5_model.keras')
    return model

model = load_trained_model()

# Define class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'] 

# Define the Streamlit app
st.title('Trash Classification App')
st.write('Upload an image of trash and the model will predict its category.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the uploaded file as bytes
    file_bytes = uploaded_file.read()
    
    # Decode the image using TensorFlow
    img = tf.io.decode_image(file_bytes, channels=3)
    img = tf.image.resize(img, [400, 400])
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Normalize the image
    img = img / 255.0
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the uploaded image
    st.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)
    
    # Display the prediction
    st.write(f'The model predicts this image is: **{predicted_class}**')
    
    # Debug: Print raw predictions and probabilities for each class
    st.write(f'Raw predictions: {predictions}')
    for class_name, probability in zip(class_names, predictions[0]):
        st.write(f'{class_name}: {probability:.4f}')
