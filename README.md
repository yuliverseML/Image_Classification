# Cats vs Dogs Image Classifier

## Project Overview
This repository contains a TensorFlow implementation of a convolutional neural network (CNN) for classifying images as either cats or dogs. The model uses transfer learning with MobileNetV2 as the base architecture, which is pre-trained on ImageNet. This approach allows for high accuracy with minimal training time.

## Features
- Transfer learning using MobileNetV2
- Data augmentation to improve model generalization
- Training/validation split for performance monitoring
- Early stopping and learning rate scheduling
- Interactive prediction on new images
- Visualization of training metrics

- **Data Preparation**:
  - Loads the dataset and splits it into training and validation sets
  - Implements preprocessing (resizing, normalization)
  - Applies data augmentation (random flips, brightness and contrast adjustments)

- **Training Process**:
  - Adam optimizer
  - Binary Cross-Entropy loss
  - Early stopping based on validation loss
  - Learning rate reduction on plateau
  - Performance monitoring with accuracy metrics

## Results
- **Validation Accuracy**: 97.2% after 15 epochs
- **Training Accuracy**: 98.5% 
- **Validation Loss**: 0.092

## Outcome
- Successfully created a robust classifier that can distinguish between cats and dogs with high accuracy
- The model demonstrates good generalization to unseen images from different sources
- Inference time is fast enough for real-time applications (~150ms per image on CPU)
- The transfer learning approach reduced training time by approximately 80% compared to training from scratch
- Augmentation techniques effectively prevented overfitting despite the limited dataset size

## Future Work
   - Experiment with different backbone architectures (EfficientNet, ResNet)
   - Implement more advanced augmentation techniques (MixUp, CutMix)
   - Test semi-supervised learning for utilizing unlabeled data

## License
This project is licensed under the MIT License - see the LICENSE file for details.


