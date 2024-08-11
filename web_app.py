import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Load the pre-trained TensorFlow model
model_path = 'E:/Volume E/Model_training/Foodvision_model.keras'
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
    'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
    'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate',
    'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse',
    'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
    'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
    'tuna_tartare', 'waffles'
]

def preprocess_image(image, img_shape=224):
    image = image.resize((img_shape, img_shape))
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the class of the image
def predict(image):
    preds = model.predict(image)
    class_idx = np.argmax(preds)
    return class_idx, preds

# Streamlit app
st.title("Image Classifier")

option = st.selectbox("Choose an option:", ("Upload Image", "Capture Photo"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        processed_image = preprocess_image(image)
        class_idx, preds = predict(processed_image)

        st.write(f"Predicted Class: {class_names[class_idx]}")
        st.write(f"Confidence: {np.max(preds):.2f}")

elif option == "Capture Photo":
    capture_button = st.button("Capture Image")
    if capture_button:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption="Captured Image", use_column_width=True)

                processed_image = preprocess_image(image)
                class_idx, preds = predict(processed_image)

                st.write(f"Predicted Class: {class_names[class_idx]}")
                st.write(f"Confidence: {np.max(preds):.2f}")
            else:
                st.write("Failed to capture image.")
        else:
            st.write("Webcam not accessible.")