# Fine-Tuning a Pre-trained EfficientNetV2B0 for 10 Food Classes

## Overview

This project uses a **pre-trained EfficientNetV2B0** model to classify images of 10 food categories. The workflow includes data preprocessing, initial training, fine-tuning, and evaluating the model's performance.

## Results

### Dataset: **10 Food Classes**
- **Source**: Balanced dataset of 10 food categories, including pasta, pizza, and others.
- **Structure**: Train and test directories with images resized to **224x224 pixels** for consistency.
- **Purpose**: Train the model to accurately classify food images.

### Model: **EfficientNetV2B0**
- **EfficientNetV2**: A state-of-the-art CNN model, pre-trained on ImageNet for improved feature extraction.
- **Transfer Learning**: The model reuses pre-trained weights for feature extraction.
- **Fine-tuning**: The last 10 layers of the model are unfrozen to adapt the model to the new food classification dataset.

### Metrics:
| Phase                     | Accuracy | Loss   |
|---------------------------|----------|--------|
| **Without Fine-tuning**    | 0.8885   | 0.3665 |
| **After Fine-tuning**      | 0.9032   | 0.2928 |

### Model Performance Visualization:
<img src="https://github.com/leovidith/EfficientNetV2B0-CNN/blob/main/Images/fine%20tuning.png" width="600">

## Features

- **Data Preprocessing**: Uses `image_dataset_from_directory` for loading data and applies data augmentation techniques such as random flips, rotations, and zooms to improve generalization.
- **Model Initialization**: The EfficientNetV2B0 model's base layers are frozen initially, and a custom head is added for classification. This consists of a `GlobalAveragePooling2D` layer followed by a `Dense` layer with a softmax activation function for multi-class classification.
- **Training Strategy**:
  - **Initial Training**: The model is first trained with the base layers frozen to avoid overfitting. Early stopping is employed to halt training once performance plateaus.
  - **Fine-tuning**: The last 10 layers of the model are unfrozen to allow for finer adjustments to the model's parameters, with a reduced learning rate to prevent catastrophic forgetting.

## Sprint Features

### Sprint 1: Data Preparation
- **Deliverable**: The dataset is loaded, augmented, and prepared for model training.

### Sprint 2: Initial Training
- **Deliverable**: EfficientNetV2B0 model is trained with frozen base layers.

### Sprint 3: Fine-tuning
- **Deliverable**: The last 10 layers are unfrozen, and the model is retrained with a reduced learning rate.

### Sprint 4: Evaluation
- **Deliverable**: The model's performance is evaluated using accuracy and loss metrics.

## Conclusion

Fine-tuning the **EfficientNetV2B0** model resulted in improved accuracy and reduced loss, demonstrating that leveraging pre-trained models with minimal additional training can be a highly effective approach for image classification tasks.
