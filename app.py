import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('./cnn-2.keras')

def predict_image(img):
    # Preprocess the image
    img = img.resize((64,64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Fill in Class Labels (i.e., what are our four classes?)
    class_labels = ['Attentive', 'DrinkingCoffee', 'UsingMirror', 'UsingRadio']
    return class_labels[predicted_class]

st.title("DISTRACTED DRIVING")

# Create your upload image button
uploaded_file = st.file_uploader("Prompt", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Make your prediction here
    prediction = predict_image(image)
    st.write(f'This image is an example of {prediction}')

    # FILL THIS IN
