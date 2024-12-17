import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# GPU Configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU found. Please ensure a GPU is available.")
    exit(1)

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    print("GPU is successfully set for TensorFlow.")
except RuntimeError as e:
    print(f"Error setting GPU: {e}")
    exit(1)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5

train_dir = 'v_data/train'
test_dir = 'v_data/test'

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['cats', 'dogs']  # Restricting to cats and dogs
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['cats', 'dogs'],  # Restricting to cats and dogs
    shuffle=False
)

# Model Definition
base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=1000,
    decay_rate=0.9
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Save Model
model.save('imagesprocessing.h5')

dense_layers = []
for layer in model.layers:
    if isinstance(layer, layers.Dense):  
        dense_layers.append(layer)

# Now, save the weights from Dense layer 1 -> Dense layer 2 -> Output layer
# Dense layer 1: 64 nodes
weights_dense1 = dense_layers[0].get_weights()[0]  # Extract weights from Dense layer 1
weights_dense1_filename = "dense1_weights.txt"
with open(weights_dense1_filename, 'w') as file:
    file.write("{\n")
    for i in range(weights_dense1.shape[0]):
        file.write('  {')
        for j in range(weights_dense1.shape[1]):
            file.write(str(weights_dense1[i][j]))
            if j != weights_dense1.shape[1] - 1:
                file.write(', ')
        file.write('}')
        if i != weights_dense1.shape[0] - 1:
            file.write(', \n')
    file.write("\n}")
print(f"Saved weights from Dense 1 in {weights_dense1_filename}")


# Dense layer 2: 32 nodes
weights_dense2 = dense_layers[1].get_weights()[0]  # Extract weights from Dense layer 2
weights_dense2_filename = "dense2_weights.txt"
with open(weights_dense2_filename, 'w') as file:
    file.write("{\n")
    for i in range(weights_dense2.shape[0]):
        file.write('  {')
        for j in range(weights_dense2.shape[1]):
            file.write(str(weights_dense2[i][j]))
            if j != weights_dense2.shape[1] - 1:
                file.write(', ')
        file.write('}')
        if i != weights_dense2.shape[0] - 1:
            file.write(', \n')
    file.write("\n}")
print(f"Saved weights from Dense 2 in {weights_dense2_filename}")

# Output layer: 2 classes
weights_output = dense_layers[2].get_weights()[0]  # Extract weights from output layer
weights_output_filename = "output_weights.txt"
with open(weights_output_filename, 'w') as file:
    file.write("{\n")
    for i in range(weights_output.shape[0]):
        file.write('  {')
        for j in range(weights_output.shape[1]):
            file.write(str(weights_output[i][j]))
            if j != weights_output.shape[1] - 1:
                file.write(', ')
        file.write('}')
        if i != weights_output.shape[0] - 1:
            file.write(', \n')
    file.write("\n}")
print(f"Saved weights from output layer in {weights_output_filename}")

# Plot Training and Validation Accuracy
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
epochs_range = range(1, len(training_accuracy) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix Plot
def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        cm[true, pred] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# Predict and Plot Image
def predict_and_plot_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print(f"Prediction probabilities: {prediction}")
    predicted_class = class_names[np.argmax(prediction)]
    print(f"Predicted class: {predicted_class}")

    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()

# Generate Class Names
class_names = {v: k for k, v in train_generator.class_indices.items()}
print(f"Class Names: {class_names}")

# Confusion Matrix
test_labels = test_generator.classes
predictions = np.argmax(model.predict(test_generator), axis=1)
plot_confusion_matrix(test_labels, predictions, list(class_names.values()))

# Example Prediction
predict_and_plot_image('v_data/test/cats/cat.4007.jpg')
