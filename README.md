# Cats vs Dogs Image Classification
A TensorFlow-based machine learning project to classify images of cats and dogs using transfer learning with MobileNetV2.

## Overview
This project demonstrates the implementation of an image classification model to distinguish between cats and dogs. It leverages TensorFlow, TensorFlow Datasets for data, and MobileNetV2 for feature extraction, followed by custom layers for classification.

### Features
- **Transfer Learning**: Utilizes MobileNetV2 pre-trained on ImageNet for feature extraction.
- **Data Handling**: Uses TensorFlow Datasets to manage the `cats_vs_dogs` dataset.
- **Image Preprocessing**: Resizes images to 224x224 pixels and normalizes pixel values.
- **Model Training**: Fine-tunes the model on a subset of the cats vs dogs dataset.
- **Prediction**: Includes functionality to upload new images and predict whether they are of cats or dogs.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Google Colab (for file upload functionality in the provided script)

## Model Architecture
- **MobileNetV2**: Base model for feature extraction, with top layers removed.
- **GlobalAveragePooling2D**: Reduces spatial dimensions of the output from MobileNetV2.
- **Dropout**: Added for regularization to prevent overfitting.
- **Dense**: A single neuron for binary classification, outputting logits.
