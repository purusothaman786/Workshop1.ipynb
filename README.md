# Binary Classification with Neural Networks on the Census Income Dataset

## Income Prediction Workshop

This repository contains a Jupyter notebook (`census_income_workshop.ipynb`) that implements a machine learning model to predict whether an individual's income exceeds **$50,000** based on the Census Income (Adult) dataset.

The project follows a structured approach, from data preparation to model evaluation.

---

# Overview

The core objective of this workshop is to build and train a neural network using **PyTorch** for a classification task on tabular data.

The notebook performs the following key steps:

---

## 1. Data Preparation

- Cleans the raw data and handles missing values
- Separates features into:
  - Categorical features
  - Continuous features
- Preprocesses the data by:
  - Encoding categorical variables
  - Standardizing continuous variables
- Converts the data into PyTorch tensors
- Splits the dataset into training and testing sets

---

## 2. Model Design

Defines a custom `TabularModel` class using PyTorch's `nn.Module`.

### Model Features
- Embeddings for categorical features
- Batch normalization for continuous features
- Hidden layers with ReLU activation
- Dropout regularization
- Output layer for binary classification

---

## 3. Training & Evaluation

The model is trained for **300 epochs** using:

- `CrossEntropyLoss`
- `Adam Optimizer`

### Evaluation Metrics
- Final Loss
- Accuracy

---

## 4. Prediction Function (Bonus)

Includes an optional prediction function that:

- Accepts new user input
- Preprocesses the input data
- Generates real-time predictions using the trained model

---

