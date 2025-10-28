# Cassava Leaf Disease Classification using EfficientNetB3

## Overview
This project applies deep learning and image classification techniques to detect different types of cassava leaf diseases from plant images.
By leveraging transfer learning with EfficientNetB3, the model aims to accurately classify infected and healthy leaves — assisting farmers in early disease detection and yield improvement.

---
## Objective

To build a robust image classification model capable of identifying cassava leaf diseases from images using pretrained CNN architectures and data augmentation techniques.

## Problem Statement

Cassava is a key food crop across tropical regions, but its yield is threatened by various leaf diseases. Manual diagnosis is time-consuming and error-prone.
This project addresses the question:

> “Can a deep learning model accurately classify cassava leaf diseases from images to support early detection and precision agriculture?”
---
## Dataset

**Source:** Kaggle - Cassava Leaf Disease Classification

**Key Files Used:**

- **train.csv** – contains image IDs and labels

- **label_num_to_disease_map.json**– maps numeric labels to disease names

- **train_images/** – folder containing training images

- **test_images/** – folder containing test images
  
**Classes:**

- **Cassava Bacterial Blight (CBB)**

- **Cassava Brown Streak Disease (CBSD)**

- **Cassava Green Mottle (CGM)**

- **Cassava Mosaic Disease (CMD)**

- **Healthy Leaves**

---

## Technologies Used

- **Python**

- **TensorFlow / Keras – Deep learning and model training**

- **Pandas / NumPy – Data manipulation**

- **Matplotlib / Seaborn – Data visualization**

- **scikit-learn – Train-validation split**

- **PIL (Pillow) – Image loading and resizing**
---

## Data Preparation & Augmentation
- **Read and mapped image labels using label_num_to_disease_map.json**

- **Visualized sample images from each class**

- **Handled class imbalance by stratified sampling**

- **Applied ImageDataGenerator for:**

- **Random rotations (up to 60°)**

    - Width & height shifts

    - Shearing and zooming

    - Horizontal & vertical flips

    - EfficientNet preprocessing for normalization

---      

## Model Architecture

- Base Model: EfficientNetB3 (pretrained on ImageNet)

- Global Average Pooling and Flatten layers

- Dense layers with ReLU activation and L1/L2 regularization

- Dropout (0.7) to prevent overfitting

- Output Layer: Softmax activation (5 classes)

- **Optimizer:** Adam (learning rate = 2e-4)
- **Loss Function:** Categorical Crossentropy with label smoothing
- **Metrics:** Categorical Accuracy
---
## Training Strategy

- **Callbacks Used:**

    - **EarlyStopping** – stop training if no improvement in validation loss

    - **ModelCheckpoint** – save best model automatically

    - **ReduceLROnPlateau** – lower learning rate on validation loss plateau

- **Epochs:** 5 (initial), fine-tuned for additional 3 epochs

- **Batch Size:** 32

- **Validation Split:** 5–10%

---

## Model Evaluation

- Plotted Training vs. Validation Accuracy and Loss Curves

- Observed stable convergence after fine-tuning with new data

- Best performing model: Cassava_best_modelEffNetB3v3.h5
---

## Predictions

- Loaded trained model for inference on unseen test images

- Resized test images to (300, 300)

- Predicted disease class using model’s softmax output

- Displayed test images with predicted labels using Matplotlib
---

## Key Insights

- Transfer learning using EfficientNetB3 achieved strong generalization on plant leaf datasets.

- Data augmentation helped reduce overfitting on limited samples.

- Model effectively distinguishes healthy leaves from multiple disease types.
---

## Business Implications

- Enables AI-driven crop disease detection, reducing manual diagnosis time.

- Can be integrated into mobile apps or agriculture advisory tools.

- Supports farmers and agricultural scientists in early disease management and yield protection.
---

## Repository Structure

- README.md

- Data

- cassava_leaf_classification.ipynb (main notebook)

- visuals/ (optional – plots and results)

---

## Author
**Udit Kaushik**  
uditkaushikk555@gmail.com  
LinkedIn Profile : (https://www.linkedin.com/in/udit-kaushik-883341377)


