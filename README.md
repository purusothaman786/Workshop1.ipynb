# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

## Income Prediction Workshop

This repository contains a Jupyter notebook (`census_income_workshop.ipynb`) that implements a machine learning model to predict whether an individual's income exceeds $50,000 based on the Census Income (Adult) dataset. The project follows a structured approach, from data preparation to model evaluation.

---

## Overview

The core objective of this workshop is to build and train a neural network using PyTorch for a classification task on tabular data. The notebook performs the following key steps:

### 1. Data Preparation

- Cleans the raw data and handles missing values.
- Separates features into categorical and continuous types.
- Preprocesses the data by:
  - Encoding categorical variables
  - Standardizing continuous variables
- Converts the data into PyTorch tensors.
- Splits the dataset into training and testing sets.

---

### 2. Model Design

Defines a custom `TabularModel` class using PyTorch's `nn.Module`.

The model:
- Incorporates embeddings for categorical features.
- Applies batch normalization for continuous features.
- Adds a hidden layer with dropout for regularization.

---

### 3. Training & Evaluation

Trains the model for 300 epochs using:
- `CrossEntropyLoss`
- `Adam` optimizer

Evaluates the performance on the test set, reporting:
- Final loss
- Accuracy

---

### 4. Prediction Function (Bonus)

Includes an optional function to allow a user to input new data.

Provides real-time predictions from the trained model.
