import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load  trained model
MODEL_PATH = 'potato_disease_classification_model.h5'
model = load_model(MODEL_PATH)

def load_and_prepare_image(image_data):
    img = load_img(image_data, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_class(img_array):
    class_names = ['Early_Blight', 'Healthy', 'Late_Blight']
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

def main():
    st.title("Potato Leaf Disease Classification")
    st.write("Upload an image of the potato leaf to classify.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        is_display = st.checkbox("Display Image")
        if is_display:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Classify'):
            # Prepare the image
            img_array = load_and_prepare_image(uploaded_file)
            # Predict the class of the image
            predicted_class, confidence = predict_image_class(img_array)
            st.success(f'Predicted Class: {predicted_class}')
            st.info(f'Confidence: {confidence:.4f}')

if __name__ == '__main__':
    main()
