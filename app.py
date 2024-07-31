import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('tuned_efficientnetb5_model.keras')
    return model

model = load_trained_model()

# Define class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'] 

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((400, 400))
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Define the Streamlit app
st.title('Trash Classification App')
st.write('Upload an image of trash and the model will predict its category.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the prediction
    st.write(f'The model predicts this image is: **{predicted_class}**')
