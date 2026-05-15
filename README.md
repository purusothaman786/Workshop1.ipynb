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

# Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

---

# Project Structure

```bash
├── census_income_workshop.ipynb
├── README.md
└── dataset/
```

---

# Dataset

This project uses the **Adult Census Income Dataset** for binary classification tasks.

---

# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/your-username/repository-name.git
```

## 2. Navigate to the project directory

```bash
cd repository-name
```

## 3. Install dependencies

```bash
pip install torch pandas numpy scikit-learn jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the notebook

```bash
census_income_workshop.ipynb
```

---

# Expected Outcome

The trained neural network predicts whether a person's income is:

- `>50K`
- `<=50K`

based on demographic and employment-related information.

---

# License

This project is open-source and available under the MIT License.
