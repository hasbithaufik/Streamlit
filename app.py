import streamlit as st

import tensorflow as tf
from tensorflow import keras

import cv2
import PIL
from PIL import Image, ImageOps

import numpy as np

import os

css = """
body {
    background-color: navy;
    color: white;
}
"""

# Apply custom CSS
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>CAT-DOG CLASSIFICATION</h1>", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)

def classifier(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (160, 160)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    
    return  prediction,prediction_percentage


uploaded_file = st.file_uploader("Choose a Cat or Dog Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,perc = classifier(image, 'model.h5')
    if label == 1:
        st.write("Its a Dog, confidence level:",perc)
    else:
        st.write("Its a Cat, confidence level:",1-perc)