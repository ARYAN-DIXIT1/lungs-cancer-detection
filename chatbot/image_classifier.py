import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# Load Pretrained Model
model = EfficientNetB0(weights="imagenet")  # Uses ImageNet weights

# Define Class Labels (Custom for Lung Cancer)
CLASS_LABELS = {
    0: "Normal",
    1: "Adenocarcinoma Lung Cancer",
    2: "Squamous Cell Lung Cancer"
}

def predict_lung_cancer(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Predict using the model
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Get highest probability class
    result = CLASS_LABELS.get(class_idx)

    return result
