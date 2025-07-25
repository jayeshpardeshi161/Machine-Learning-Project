# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, and several methods such as **SMOTE oversampling**, **Random Forest classification**, and **data preprocessing pipelines** were applied to build a robust fraud detection system. A **Streamlit web application** is also included for real-time prediction and demonstration.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Dataset Overview](#dataset-overview)
- [Approach](#approach)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)

---

## ❓ Problem Statement

Credit card fraud is a growing concern in the financial world. Fraudulent transactions are rare but cause significant financial losses. The challenge lies in accurately identifying fraud while minimizing false positives and negatives in a heavily imbalanced dataset.

---

## 🎯 Project Goals

- Build an accurate machine learning pipeline for binary classification (Fraud vs. Normal)
- Handle data imbalance effectively using SMOTE
- Deploy the model using Streamlit for interactive predictions

---

## 📊 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Features**:
  - `V1` to `V28`: PCA-transformed features (anonymized)
  - `Amount`: Transaction amount
  - `Time`: Time since first transaction
  - `Class`: Target variable (0 = Normal, 1 = Fraud)

---

## 🧪 Approach

1. Load and explore the dataset
2. Perform data preprocessing and check for missing values
3. Analyze class imbalance and apply SMOTE to balance the training set
4. Train a `RandomForestClassifier` within a `Pipeline` (including scaling)
5. Evaluate the model using ROC AUC and classification metrics
6. Save the trained model with `joblib`
7. Build and deploy a Streamlit app for real-time predictions

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution of normal vs. fraud transactions
- Heatmap of feature correlations
- Boxplot analysis of `Amount` per class
- Feature inspection and identification of outliers

---

## 🤖 Modeling

- **Model**: Random Forest Classifier
- **Pipeline Steps**:
  - `StandardScaler` for feature normalization
  - `SMOTE` for oversampling the minority class
  - Random Forest with 100 trees

---

## 📈 Evaluation Metrics

- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: True/False Positives/Negatives
- **ROC AUC Score**: Evaluates model’s class separation capability

---

## 🔗 License

MIT License © 2025 [Jayesh Pardeshi]

---

## 🔗 Contact

📧 Gmail	:[jayeshpardeshi161@gmail.com]  
📌 LinkedIn:[] 
📌 Portfolio:[]

---
## 🚀 Streamlit App

An interactive web application built using **Streamlit**.

### 🔧 Features:
- Manual input or use sample transactions
- Real-time prediction of transaction status
- Fraud probability display
- Model caching and feature validation


### 📁 File: `app.py`

### To run the app locally:

####  ```bash
streamlit run app.py

---



