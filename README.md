# Fine-Tuning a Pre-trained EfficientNetV2B0 for 10 Food Classes

## Overview
This project uses a **pre-trained EfficientNetV2B0** model for classifying images of 10 food classes. The process includes data preprocessing, initial training, fine-tuning, and evaluation.

---

## Dataset: **10 Food Classes**
- **Source**: Contains balanced images of 10 food categories (e.g., pasta, pizza).  
- **Structure**: Train and test directories with images resized to **224x224 pixels**.  
- **Purpose**: Train the model to classify food accurately.

---

## Model: **EfficientNetV2B0**
- **EfficientNetV2**: A state-of-the-art CNN, pre-trained on ImageNet.  
- **Transfer Learning**: Reuses pre-trained weights for feature extraction.  
- **Fine-tuning**: Last 10 layers unfrozen to adapt features to the new dataset.

---

## Workflow
1. **Data Preprocessing**:
   - Load data with `image_dataset_from_directory`.
   - Apply data augmentation (random flips, rotations, zooms).  

2. **Model Initialization**:
   - Base EfficientNet layers frozen.
   - Custom head: GlobalAveragePooling2D + Dense (softmax) for classification.

3. **Training**:
   - **Initial Training**:
     - Base layers frozen.
     - Early stopping to avoid overfitting.
   - **Fine-tuning**:
     - Last 10 layers unfrozen.
     - Reduced learning rate for gradual adaptation.

---

## Metrics
| Phase              | Accuracy | Loss   |
|--------------------|----------|--------|
| **Without Fine-tuning** | 0.8885   | 0.3665 |
| **After Fine-tuning**   | 0.9032   | 0.2928 |

<img src="https://github.com/leovidith/EfficientNetV2B0-CNN/blob/main/Images/fine%20tuning.png" width="600">

---

## Conclusion
Fine-tuning improved accuracy and reduced loss, demonstrating the effectiveness of leveraging pre-trained models with minimal additional training.
