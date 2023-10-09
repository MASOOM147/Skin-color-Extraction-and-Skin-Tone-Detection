# Skin Color Extraction and Skin Tone Detection

This repository contains code for skin color extraction and skin tone detection using Python and various libraries. The project is divided into several sections to accomplish different tasks related to skin color analysis.

## Table of Contents
- [Setup](#setup)
- [Skin Color Extraction](#skin-color-extraction)
- [Skin Tone Detection](#skin-tone-detection)
- [Test](#test)
- [Pre-Trained Model](#pre-trained-model)

---

## Setup

Before running the code, ensure that you have the necessary libraries installed. You can run this code in a Jupyter Notebook or any Python environment.

1. Mount Google Drive: If you are using Google Colab, start by mounting your Google Drive to access the required data and models.

2. Data Preparation: The code expects labeled image data for skin color analysis. Make sure you have your data structured appropriately.

3. Libraries: The code utilizes various libraries such as OpenCV, NumPy, TensorFlow, and scikit-learn. Ensure these libraries are installed in your environment.

---

## Skin Color Extraction

The "Skin Color Extraction" section focuses on extracting skin color pixels from an image using predefined criteria. It includes the following steps:

1. **Loading the Image**: Load an image on which you want to perform skin color extraction.

2. **Skin Color Detection**: Implement a function to detect skin color pixels based on predefined criteria.

3. **Display the Skin Mask**: Display the mask where white pixels represent skin tone color.

4. **Visualization**: Visualize the skin mask to observe the detected skin color regions in the image.

---

## Skin Tone Detection

The "Skin Tone Detection" section involves training a deep learning model to classify skin color into predefined categories. It includes the following steps:

1. **Dataset Preparation**: Load and preprocess the labeled dataset for skin tone classification. The dataset includes images categorized into different skin tone classes.

2. **Data Augmentation**: Optionally, apply data augmentation techniques to increase the dataset's diversity.

3. **Model Building**: Construct a convolutional neural network (CNN) model for skin tone classification.

4. **Model Training**: Train the model using the prepared dataset and monitor its performance.

5. **Evaluation**: Evaluate the trained model on a validation set and calculate accuracy.

6. **Model Saving**: Save the trained model for future use.

7. **Prediction**: Make predictions using the trained model on new images to classify skin tone categories.

---

## Test

The "Test" section includes code for testing skin tone within a specified range using an example Region of Interest (ROI). It includes the following steps:

1. **Loading and Preprocessing**: Load an image, convert it to RGB, HSV, and YCbCr color spaces, and define an ROI.

2. **Skin Color Analysis**: Calculate the mean hue value within the ROI and check if it falls within the specified skin tone range.

3. **Visualization**: Visualize the ROI and indicate whether the image contains a face with a skin tone within the range.

---

## Pre-Trained Model

The "Pre-Trained Model" section demonstrates the use of a pre-trained MobileNetV2 model for skin tone classification. It includes the following steps:

1. **Data Augmentation**: Apply data augmentation techniques to the dataset to enhance model performance.

2. **Model Building**: Load a pre-trained MobileNetV2 model and add layers for skin tone classification.

3. **Model Training**: Compile and train the model on the prepared dataset.

4. **Evaluation**: Evaluate the model's performance on the test set and report accuracy.

5. **Prediction**: Make predictions on new images using the trained model.

Feel free to explore and run the code sections as needed for your skin color analysis tasks. Make sure to replace image paths and dataset locations with your specific data.

For any questions or issues, refer to the code comments and documentation, or reach out for assistance.
helu...
**Note**: This README provides an overview of the code's functionality and usage. Further details and code execution are required to perform skin color extraction and skin tone detection effectively.
