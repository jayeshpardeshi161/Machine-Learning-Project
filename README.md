# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, and several methods such as undersampling, oversampling (SMOTE), and different classifiers were applied to build a reliable fraud detection system.

## 🔍 Problem Statement
Credit card fraud is a growing problem in the financial world. The goal of this project is to accurately classify transactions as fraudulent or legitimate.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - 30 total features (`V1` to `V28`, `Amount`, `Time`)
  - `Class` column: `0` = Normal, `1` = Fraud

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)
- Streamlit (for model deployment)
- Joblib (for model serialization)

---

## 🧪 Model Building Steps

1. **Data Preprocessing**
   - Scaled `Amount` feature using `StandardScaler`
   - Dropped `Time` column and removed duplicate rows

2. **Class Imbalance Handling**
   - Visualized imbalance (`Class` 0 >> `Class` 1)
   - Applied:
     - **Undersampling** (limited normal transactions)
     - **Oversampling** with **SMOTE**

3. **Model Training & Evaluation**
   - Trained `Logistic Regression` and `Decision Tree` classifiers
   - Evaluated using:
     - Accuracy
     - Precision
     - Recall
     - F1 Score

4. **Model Deployment**
   - Final model saved using `joblib`
   - Built an interactive **Streamlit app** for real-time predictions

---

## 📊 Results

| Model               | Precision | Recall | F1 Score |
|--------------------|-----------|--------|----------|
| Logistic Regression | 0.94      | 0.88   | 0.91     |
| Decision Tree       | 0.96      | 0.92   | 0.94     |

> *(Metrics may vary slightly depending on sampling and random state)*

---

## 🚀 Streamlit App

Run locally:

```bash
streamlit run app.py
Input:
29 transaction feature values (V1 to V28 + Amount)

Output:
"✅ Normal Transaction" or "⚠️ Fraudulent Transaction"

🧠 Learnings
Handling highly imbalanced datasets

Model evaluation beyond just accuracy

Real-world deployment with Streamlit

Importance of precision & recall in fraud detection

📦 Folder Structure
bash

credit-card-fraud-detection/
│
├── app.py                     # Streamlit web app
├── creditcard.csv             # Dataset (not uploaded to GitHub)
├── credit_card_model.pkl      # Trained model
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies


---

✅ 1. requirements.txt

streamlit
joblib
numpy
scikit-learn

✅ 2. README.md

# 💳 Credit Card Fraud Detection App

This is a Streamlit-based web application that uses a machine learning model to detect fraudulent credit card transactions. Enter transaction values (V1–V28 and Amount) to predict whether the transaction is **Fraudulent** or **Normal**.

---

## 🚀 Features

- Takes 29 input features: V1 to V28 + Amount
- Uses a trained ML model (`credit_card_model.pkl`)
- Provides real-time prediction
- Built with Python and Streamlit

---

## 🛠️ How to Run the App

 Install Dependencies
bash
pip install -r requirements.txt

The app will open in your browser at http://localhost:8501.

📁 Project Structure

credit_card_fraud_app/
├── app.py
├── credit_card_model.pkl
├── requirements.txt
└── README.md

📦 Dependencies

streamlit

joblib

numpy

scikit-learn

🧠 Model Info
The model (credit_card_model.pkl) is a pre-trained machine learning model using scikit-learn. If you don't have it, you need to train and export one.


## 🔗 Author

  
📧 jayeshpardeshi161@gmail.com
📌 LinkedIn: [Profile URL]  
