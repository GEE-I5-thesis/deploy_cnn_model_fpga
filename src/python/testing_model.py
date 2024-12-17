import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('imagesprocessing.h5')

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class labels for two classes: cat and dog
class_labels = {0: 'cat', 1: 'dog'}

def predict_and_plot_image(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()

    print(f"Predicted class: {predicted_class}")

# Path to the input image
input_image_path = 'v_data/test/cats/cat.4005.jpg'  # Replace this with the path to your test image

# Make a prediction
predict_and_plot_image(input_image_path)
