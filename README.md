# Cats vs Dogs Image Classifier with MobileNetV2

## Project Overview
This repository contains a TensorFlow implementation of a convolutional neural network (CNN) for classifying images as either cats or dogs. The dataset used is the [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Cats_vs_Dogs_Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).  The model uses transfer learning with MobileNetV2 as the base architecture, which is pre-trained on ImageNet. This approach allows for high accuracy with minimal training time.

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

## NOTES: Alternative Applications for This Model Across Different Domains
Suitable Applications:

    Medical Image Classification:
        Skin lesion classification (benign vs. malignant)
        X-ray analysis (normal vs. pneumonia)
        Retinal disease detection
        Advantages: Binary classification structure, transfer learning benefits, similar image processing needs

    Agricultural Monitoring:
        Crop disease detection
        Pest identification
        Fruit ripeness assessment
        Advantages: Visual pattern recognition, adaptable to different species, lightweight for field deployment

    Quality Control in Manufacturing:
        Defect detection (defective vs. non-defective parts)
        Product verification
        Material classification
        Advantages: Binary decision-making, consistent imaging conditions, fast inference time

    Security and Surveillance:
        Weapon detection
        Unauthorized access detection
        Suspicious behavior identification
        Advantages: Real-time capabilities, mobile deployment options, binary classification approach

Potentially Suitable Applications:

    Satellite and Aerial Imagery Analysis:
        Land use classification
        Disaster damage assessment
        Urban development monitoring
        Considerations: May need different input dimensions, different color channels, larger context windows

    Document Processing:
        Signature verification
        Document type classification
        Handwriting analysis
        Considerations: Requires fine-tuning for text/document features, different aspect ratios

    Retail and E-commerce:
        Product categorization
        Counterfeit detection
        Visual search features
        Considerations: May need multi-class extensions, different image preprocessing

    Traffic Monitoring:
        Vehicle type classification
        License plate detection
        Traffic density analysis
        Considerations: Requires handling motion blur, different lighting conditions, occlusions

## License
This project is licensed under the MIT License - see the LICENSE file for details.


