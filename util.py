import base64
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


def classify(img, model, class_names, unknown_threshold=50):
    # Preprocess the image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class and confidence
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Check if confidence is below the unknown threshold
    if confidence < unknown_threshold:
        predicted_class = "Unknown"

    return predicted_class, confidence
