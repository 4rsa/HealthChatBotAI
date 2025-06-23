import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def build_brain_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=4):
    """
    Builds the CNN model architecture. Must match the original architecture
    used during training.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
   
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build the model (same architecture as training)
model = build_brain_model()

# Load weights
weights_path = os.path.join(os.path.dirname(__file__), 'storage', 'brain.weights.h5')
model.load_weights(weights_path)

def preprocess_image(image_path):
    """
    Loads and preprocesses an image from disk for model prediction.
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads as BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Expand dimensions so we have shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image_path):
    """
    Classifies an image and returns the predicted class name.
    """
    # Preprocess image
    preprocessed_img = preprocess_image(image_path)
    
    # Perform prediction
    predictions = model.predict(preprocessed_img)
    
    # predictions is a 2D array; for a single image, index 0
    predicted_idx = np.argmax(predictions[0])
    
    # Return the class name
    return CLASS_NAMES[predicted_idx]
