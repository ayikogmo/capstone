import streamlit as st
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Set title and header
def main():
    st.title('KutisNet')
    st.subheader('A CNN-based Skin Disease Image Recognition')
    st.markdown('**Disclaimer:** The KutisNet model is designed and trained specifically for the recognition and classification of dermatological conditions based on images of skin. As such, the accuracy and reliability of its predictions are limited to images that depict skin conditions, specifically **warts**, **psoriasis**, **normal skin**, **eczema**, and **acne**.')
    st.markdown("**Please note:** Any images that do not represent dermatological conditions, such as images of non-related subjects, objects , or backgrounds, will likely result in incorrect predictions. The model is not equipped to interpret or analyze images outside of its intended scope, and users should refrain from submitting such images for analysis.")
    st.markdown('**Please upload an image**')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result, confidence = predict_class(image)
        st.title(result)
        st.markdown(f"**Confidence:** {confidence}%")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)


def predict_class(image):
    loaded_model = keras.models.load_model('model.h5', compile=False)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)  # Resize the image to match the input size of the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = loaded_model.predict(image)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    class_labels = ['Acne', 'Eczema', 'Normal', 'Psoriasis', 'Warts']  # Define your class labels
    predicted_class_label = class_labels[predicted_class_index]

    # Get the highest value in the predictions array
    highest_value = round(np.max(predictions) * 100, 2)
    return predicted_class_label, highest_value

if __name__ == '__main__':
    main()
