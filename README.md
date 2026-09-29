# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Overview

Credit Card Fraud Detection is a machine learning project that identifies whether a credit card transaction is legitimate or fraudulent.

The project uses Logistic Regression with feature scaling and class balancing to handle the highly imbalanced transaction dataset.

An interactive Streamlit web application is developed to demonstrate transaction classification, fraud probability, and risk assessment.

## 🎯 Objectives

- Detect fraudulent credit card transactions using Machine Learning.
- Perform data cleaning and preprocessing.
- Handle highly imbalanced transaction classes.
- Train and evaluate a Logistic Regression model.
- Display fraud probability for a transaction.
- Provide an interactive Streamlit interface.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 🤖 Machine Learning Approach

### Model

**Logistic Regression**

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

### Preprocessing

- Duplicate removal
- Feature-target separation
- Train-test split
- StandardScaler feature scaling
- Class balancing using `class_weight="balanced"`

## 📊 Dataset

The dataset contains anonymized credit card transactions.

Features include:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `Class` column is the target variable.

The dataset is highly imbalanced, with fraudulent transactions representing a very small portion of all transactions.

## 🔍 Exploratory Data Analysis

The project includes analysis of:

- Legitimate vs fraudulent transactions
- Transaction amount distribution
- Transaction amount by class
- Transaction time distribution
- Fraudulent transaction time distribution
- Duplicate transactions

## 🖥️ Streamlit Application

The application provides:

- Transaction selection
- Fraud/legitimate prediction
- Fraud probability
- Risk assessment
- Model information
- Interactive dashboard

### Application Workflow

```text
Transaction
     ↓
Preprocessing
     ↓
Feature Scaling
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Fraud Probability
     ↓
Risk Assessment

### Run the application ###
python -m streamlit run app.py

The application will open at:

http://localhost:8501
📈 Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

Because the dataset is highly imbalanced, accuracy alone is not sufficient for evaluating fraud detection performance.

🚀 Future Improvements
Add Random Forest and other machine learning models.
Add interactive transaction input.
Improve fraud-risk visualization.
Add model comparison.
Add explainable AI techniques.
Deploy the application online.
👨‍💻 Author

Vansh Gupta

B.Tech Computer Science & Engineering

📌 Disclaimer

This project is developed for educational and demonstration purposes. It is not intended to make real-world financial or banking decisions

