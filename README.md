# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. 
The dataset used is highly imbalanced, and several methods such as undersampling, oversampling (SMOTE), and different classifiers were applied to build a reliable fraud detection system.
Built an interactive **Streamlit app** for real-time predictions with Python.

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

---

## 🧠 Learnings
Handling highly imbalanced datasets

Model evaluation beyond just accuracy

Real-world deployment with Streamlit

Importance of precision & recall in fraud detection

---

## 📦 Folder Structure
bash

credit-card-fraud-detection/
│
├── app.py                     # Streamlit web app
├── creditcard.csv             # Dataset (not uploaded to GitHub)
├── credit_card_model.pkl      # Trained model
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies


---


## ✅ app.py Code

Python
# app.py

import streamlit as st
import numpy as np
import joblib

# Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details below to check if it's **Fraudulent or Normal**")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

model = load_model()

# Input features
features = []
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

st.subheader("📝 Transaction Details Input")
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0, format="%.5f")
    features.append(val)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict([features])[0]
    if prediction == 0:
        st.success("✅ This is a **Normal Transaction**")
    else:
        st.error("⚠️ This is a **Fraudulent Transaction**")

---

## ✅ 1. requirements.txt

streamlit
joblib
numpy
scikit-learn

---

## ✅ 2. README.md

# 💳 Credit Card Fraud Detection App

This is a Streamlit-based web application that uses a machine learning model to detect fraudulent credit card transactions. Enter transaction values (V1–V28 and Amount) to predict whether the transaction is **Fraudulent** or **Normal**.

---

## 🚀 Features

- Takes 29 input features: V1 to V28 + Amount
- Uses a trained ML model (`credit_card_model.pkl`)
- Provides real-time prediction
- Built with Python and Streamlit

---

🔧 How to Use This App
Step 1: Save Files
Save the above code as app.py.

Ensure credit_card_model.pkl is in the same folder.

Step 2: Install Required Libraries (if not already installed)
In terminal (Command Prompt / Anaconda Prompt):

bash

pip install streamlit joblib numpy
Step 3: Run the App
bash

streamlit run app.py
It will open a browser window automatically at:

http://localhost:8501

Or

## 🛠️ How to Run the App

 Install Dependencies
bash
pip install -r requirements.txt

Run the App
bash

streamlit run app.py
The app will open in your browser at http://localhost:8501.

---

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

---

***✅ Jupyter Code***

python

## 📌 Step-by-step Explanation ( Q kiya? with reasons ):

**1. Dataset Load & Initial Exploration**

import pandas as pd
data = pd.read_csv("creditcard.csv")
data.head()

🔹 Q kiya? Dataset ko pandas ke through load kiya aur head() se initial rows dekhe taki data ka structure samajh sakein.

pd.options.display.max_columns = None
data.tail()

🔹 Q kiya? Saare columns properly dekh sakein, isliye max_columns None kiya, aur tail() se last rows dekhi.


data.shape
print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[0]))

🔹 Q kiya? Dataset ke size (rows & columns) ko samajhne ke liye.

data.info()
data.isnull().sum()

🔹 Q kiya? Data types aur null values check karne ke liye. Ye ensure karta hai ki missing values hain ya nahi.

**2. Feature Scaling**

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
🔹 Q kiya? Amount column ko scale kiya ja raha hai kyunki machine learning models scale-sensitive hote hain.

**3. Drop Unnecessary Column**

data = data.drop(['Time'], axis=1)

🔹 Q kiya? 'Time' column model ke liye relevant nahi tha, isliye remove kiya.

**4. Duplicate Check & Removal**

data.duplicated().any()
data = data.drop_duplicates()

🔹 Q kiya? Duplicate rows prediction ko mislead kar sakti hain, isliye unhe remove kiya.

**5. Class Imbalance Analysis**

data['Class'].value_counts()

🔹 Q kiya? Fraudulent vs legitimate transaction ka distribution dekhne ke liye. Isse imbalance ka idea milta hai.

sns.countplot(data['Class'])
plt.show()

🔹 Q kiya? Visual check kiya imbalance ko plot kar ke.

**6. Data Split for Training**

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

🔹 Q kiya? Features aur label ko alag kiya aur data ko training/testing sets me split kiya.

**7. Initial Model Training (Imbalanced Data)**

classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

🔹 Q kiya? Do different models ka comparison karne ke liye.

for name, clf in classifier.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy, precision, recall, f1 score)

🔹 Q kiya? Models ko train kiya aur evaluate kiya unke performance metrics se.

**8. Undersampling (Class Balance Karna)**

normal = data[data['Class']==0]
fraud = data[data['Class']==1]
normal_sample = normal.sample(n=473)
new_data = pd.concat([normal_sample, fraud], ignore_index=True)

🔹 Q kiya? Dataset me se legitimate transactions ka ek chhota subset liya taaki fraud aur non-fraud ka balance ho sake (undersampling).

**9. Model Training on Undersampled Data**

X_train, X_test, y_train, y_test = train_test_split(...)
# Same training loop

🔹 Q kiya? Balanced data par model train karne se model fairness improve hoti hai.

**10. Oversampling using SMOTE**

from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)

🔹 Q kiya? Minority class (fraud) ke synthetic samples generate kiye taaki imbalance ko fix kiya ja sake.

**11. Model Training on Oversampled Data**

🔹 Q kiya? SMOTE ke baad models ko dobara train kiya taaki better accuracy mil sake.

**12. Model Saving**

dtc = DecisionTreeClassifier()
dtc.fit(X_res, y_res)
joblib.dump(dtc, "credit_card_model.pkl")

🔹 Q kiya? Trained model ko save kiya future use ke liye, bina dobara train kiye.

**13. Prediction on New Sample**

model = joblib.load("credit_card_model.pkl")
pred = model.predict(df_input)

🔹 Q kiya? Naye transaction pe prediction lene ke liye trained model ko load kiya.

if pred[0] == 1:
    print("Fraud")
else:
    print("Legit")

🔹 Q kiya? Predict kiya ki transaction fraudulent hai ya nahi.

---

## ✅ Summary:

Ye pura process kr ke credit card fraud detection ke liye machine learning model banaya, train kiya, aur optimize kar ke –
imbalance handle karte hue, models compare karke, aur best model ko save karke prediction Kiya.

---

## 📜 License
This project is licensed under the MIT License.

---

## 🔗 Author

📧 Gmail:[jayeshpardeshi161@gmail.com]
📌 LinkedIn: [Profile URL]  
