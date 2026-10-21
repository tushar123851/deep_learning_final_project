# Deep Learning, Computer Vision & NLP — Project Summary

## Overview

This portfolio contains five practical projects covering deep learning, computer vision, time-series forecasting, and natural language processing (NLP). Built with Python and widely used machine-learning libraries, the projects demonstrate a structured workflow: prepare data, apply a suitable model or image-processing technique, evaluate outputs, and interpret results. The notebooks are intended to support hands-on learning and provide clear documentation of experiments.

## Projects and Results

### 1. ANN Diabetes Prediction
An Artificial Neural Network (ANN) was trained for binary classification using the Pima Indians Diabetes dataset. The workflow includes handling implausible zero values in selected medical measurements, splitting data into training and test sets, scaling numeric features, and evaluating predictions. The reported test accuracy was **71.43%**, with **60.87% precision**, **51.85% recall**, and an **F1-score of 56.00%**. These metrics show why accuracy should be considered alongside recall and false-negative counts. This exercise is educational only and must not be used for medical decisions.

### 2. OpenCV Image Processing
This project applies an image-processing pipeline to five sample images. OpenCV is used to resize images, convert them to grayscale, reduce noise with Gaussian blur, detect edges with Canny, and identify contours. A visualization grid can display the original image alongside the processed stages, making the effect of each operation easier to understand.

### 3. CNN CIFAR-10 Classification
A Convolutional Neural Network (CNN) classifies CIFAR-10 images into ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The reported test accuracy was **60.25%** on **4,000 images**. Results varied across classes, demonstrating the importance of reviewing per-class precision, recall, and F1-score in addition to overall accuracy.

### 4. SimpleRNN Airline Passenger Forecasting
A Simple Recurrent Neural Network (SimpleRNN) forecasts monthly airline passenger counts using historical observations. The workflow scales values, creates sequences from the previous 12 months, and splits data chronologically to preserve time order. The reported test RMSE was **40.57 passengers**. Comparing predictions with actual counts and a simple seasonal baseline can help assess the model's practical value.

### 5. Transformer Sentiment Analysis
A pre-trained Hugging Face model, `distilbert-base-uncased-finetuned-sst-2-english`, classifies text as positive or negative. The exercise evaluates 25 example sentences, including positive, negative, neutral, and ambiguous text. Since the model is binary, neutral sentences are still assigned one of the two labels. Confidence scores reflect the model's preference and do not guarantee that a prediction is correct.

## Technology Stack and Learning Outcomes

The portfolio uses Python, NumPy, Pandas, Matplotlib, OpenCV, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, and Jupyter Notebook or VS Code. The projects build practical experience in data preprocessing, feature scaling, neural-network training, image processing, time-series sequence creation, sentiment classification, and model evaluation.

## Future Improvements

Possible improvements include tuning ANN thresholds and class weights, testing OpenCV settings on varied images, improving CNN performance with data augmentation and regularization, comparing SimpleRNN forecasts with seasonal baselines, and evaluating sentiment predictions on a larger labeled dataset. Recording package versions, random seeds, data sources, and genuine output screenshots will improve reproducibility.

## Author

**Tushar Vala**  
*Data Analyst | AI/ML Developer | Data Scientist*

## This portfolio reflects a commitment to practical learning, experimentation, and well-documented AI projects. Reported metrics describe specific runs and may vary with changes to data, preprocessing, random seeds, and model settings.
"""
