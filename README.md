# product-image-classifier
# Product Image Classifier

## Overview
This project implements a product image classifier for an e-commerce app. The classifier automatically categorizes product images into predefined groups such as fashion, nutrition, accessories, etc.

## Approach
- **Data Preparation**: The dataset is organized into folders, each representing a different category. Images are labeled, preprocessed, and split into training, validation, and test sets.
- **Model Building**: A Convolutional Neural Network (CNN) model is constructed using TensorFlow and Keras. Transfer learning with a pre-trained ResNet50 model is utilized for feature extraction.
- **Training**: The model is trained on the training data using the Adam optimizer and categorical cross-entropy loss function. Model performance is monitored using validation data.
- **Testing**: The trained model's performance is evaluated on a separate test set to assess its accuracy and generalization capability.
- **Submission**: The trained model is suitable for deployment to the e-commerce app's backend for automatic image categorization.

## Functionality
- **Labeling the Data**: Images are labeled based on the folders they belong to. A dictionary maps category names to numerical labels.
- **Image Preprocessing**: Images are resized to a consistent size, converted to arrays, and normalized to ensure uniform dimensions and format.
- **Splitting the Dataset**: The dataset is divided into training, validation, and test sets to facilitate model training and evaluation.

## Usage
1. **Dataset Preparation**: Organize your dataset into folders where each folder represents a category.
2. **Data Processing**: Use the provided script or code to label the data, preprocess the images, and split the dataset into training, validation, and test sets.
3. **Model Training**: Train the model using the prepared dataset and the provided model building code. Fine-tune the hyperparameters as needed.
4. **Model Testing**: Test the trained model using the provided code for model evaluation and prediction. Verify its accuracy and performance.

## Dependencies
- Python 3.x
- TensorFlow
- NumPy
- OpenCV (cv2)
- Matplotlib

## Author
[nourhan reda abdelraouf]


## video
https://drive.google.com/file/d/1ogRwPCTPbmkMrzjcoc2xPkxMcocOmMGW/view?usp=drive_link
