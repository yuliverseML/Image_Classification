# Import necessary libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from google.colab import files

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Load the cats vs dogs dataset and split into train and validation
train_ds, val_ds = tfds.load(
    'cats_vs_dogs', 
    split=[f'train[:{int(100*(1-VALIDATION_SPLIT))}%]', f'train[{int(100*(1-VALIDATION_SPLIT))}%:]'], 
    as_supervised=True
)

# Function to preprocess images (resize and normalize)
def preprocess_image(img, label):
    """
    Resize the image to target size and normalize pixel values to [0,1].
    
    Args:
        img: Input image tensor
        label: Image label
    
    Returns:
        Preprocessed image and label
    """
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1]
    return img, label

# Function to apply data augmentation
def augment_image(img, label):
    """
    Apply random augmentations to training images.
    
    Args:
        img: Input image tensor
        label: Image label
    
    Returns:
        Augmented image and label
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

# Apply preprocessing to datasets
train_ds = train_ds.map(preprocess_image).map(augment_image)
val_ds = val_ds.map(preprocess_image)

# Create batched datasets
train_batches = train_ds.shuffle(1000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_batches = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers to prevent them from being trained
base_model.trainable = False

# Build the complete model
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1)  # Single output for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )
]

# Train the model
history = model.fit(
    train_batches,
    validation_data=val_batches,
    epochs=20,
    callbacks=callbacks
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to predict on new images
def predict_image(image_path):
    """
    Make predictions on a new image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Prediction probability and display result
    """
    try:
        # Load and preprocess the image
        img = load_img(image_path)
        img_array = img_to_array(img)
        img_resized, _ = preprocess_image(img_array, 0)
        img_expanded = np.expand_dims(img_resized, axis=0)
        
        # Make prediction
        prediction = model.predict(img_expanded)
        
        # Apply sigmoid to convert logits to probability
        probability = tf.nn.sigmoid(prediction).numpy()[0][0]
        
        # Display the result
        plt.figure()
        plt.imshow(img)
        label = 'Dog' if probability > 0.5 else 'Cat'
        plt.title(f'{label} (probability: {probability:.2f})')
        plt.show()
        
        return probability
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Upload and predict on new images
print("Upload images for prediction:")
uploaded = files.upload()
image_paths = list(uploaded.keys())

for img_path in image_paths:
    predict_image(img_path)

# Optional: Save the model
model.save('cats_vs_dogs_classifier.h5')
print("Model saved successfully!")
