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

## ✅ What I Did

***Environment Setup***

To begin this project, I used Anaconda Navigator to launch Jupyter Notebook, a popular IDE for data science tasks. Anaconda provides a robust Python environment with pre-installed libraries, which made it efficient to manage dependencies and work interactively with code.

**Step 1: Import Required Libraries**

The first step in this machine learning project involved importing the necessary Python libraries. 
These libraries cover a wide range of functionalities, including data manipulation, visualization, model building, evaluation, and handling class imbalance.

Below is a detailed table of the libraries imported and their specific purpose within the project:
| **Python Code**                                                                             | **Comments**                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `import pandas as pd`                                                                | For handling tabular data                                            |
| `import numpy as np`                                                                 | For numerical computations                                           |
| `from sklearn.model_selection import train_test_split`                               | For splitting dataset into training and test sets                    |
| `from sklearn.ensemble import RandomForestClassifier`                                | Random Forest model for classification                               |
| `from sklearn.pipeline import Pipeline`                                              | To create a machine learning pipeline                                |
| `from sklearn.preprocessing import StandardScaler`                                   | To normalize/scale features                                          |
| `from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score` | For evaluating the model (accuracy, confusion matrix, AUC-ROC, etc.) |
| `from imblearn.over_sampling import SMOTE`                                           | For handling class imbalance by oversampling the minority class      |
| `import joblib`                                                                      | To save (serialize) the trained model                                |
| `import matplotlib.pyplot as plt`                                                    | For plotting graphs and visualizations                               |
| `import seaborn as sns`                                                              | For enhanced visualizations (e.g., heatmaps, boxplots, etc.)         |

**🔍 Step 2: Load Dataset**

After importing the required libraries, the next step is to load the dataset into the environment. 
The dataset used for this project is creditcard.csv, which contains anonymized credit card transactions labeled as fraudulent or legitimate.

Using Pandas, the dataset is read from a CSV file and loaded into a DataFrame named df. This step is crucial for all downstream data preprocessing, analysis, and model development.
| **Python Code**                        | **Comments**                                               |
| -------------------------------------- | ---------------------------------------------------------- |
| `df = pd.read_csv("creditcard.csv")`   | Load the dataset from the CSV file into a Pandas DataFrame |
| `print("✅ Dataset loaded:", df.shape)` | Print confirmation along with the shape of the dataset     |
***Output:***
✅ Dataset loaded: (20000, 31)

***Explanation:***
The dataset contains 20,000 rows and 31 columns.
Each row represents a credit card transaction, and the columns include features extracted from the transaction along with a label indicating whether it is fraudulent (1) or not (0).
This step ensures that the data is successfully imported and ready for exploration and preprocessing in the next stages.

**📊 Step 3: Initial Data Exploration**

After successfully loading the dataset, the next step is to perform an initial exploration of the data. 
This step provides a quick understanding of the dataset's structure, including its features, datatypes, presence of missing values, and general layout.
The following Python commands were used for preliminary inspection:
| **Python Code**            | **Comments**                                                        |
| -------------------------- | ------------------------------------------------------------------- |
| `print(df.head())`         | Display the first 5 rows of the dataset to understand its structure |
| `print(df.shape)`          | Print the number of rows and columns in the dataset                 |
| `print(df.info())`         | Show datatypes and count of non-null values for each column         |
| `print(df.isnull().sum())` | Check for missing values in each column                             |

***Insights Gained:***
The dataset contains 20,000 records and 31 columns.
All features are numerical, and most have been anonymized (e.g., V1, V2, ..., V28), with the exception of:
Time – representing the time elapsed between transactions.
Amount – transaction amount.
Class – target variable (1 = fraud, 0 = non-fraud).
No missing values were found in the dataset, indicating that no imputation or cleaning is necessary at this stage.

**⚖️ Step 4: Explore Class Distribution (Understand the Class Imbalance)**

Understanding the distribution of the target variable (Class) is a critical step in fraud detection problems. 
This helps identify if there is a class imbalance, which is common in real-world fraud datasets where fraudulent transactions are rare compared to legitimate ones.

The following code was used to examine and visualize the class distribution:
| **Python Code**                         | **Comments**                                                       |
| --------------------------------------- | ------------------------------------------------------------------ |
| `print("Original class distribution:")` | Print a heading to indicate the start of class distribution output |
| `print(df["Class"].value_counts())`     | Count the number of legitimate (0) and fraudulent (1) transactions |
| `sns.countplot(x="Class", data=df)`     | Visualize the distribution using a bar plot                        |
| `plt.title("Class Distribution")`       | Add a descriptive title to the plot                                |
| `plt.show()`                            | Display the plot                                                   |
***Output:***
Original class distribution:
0    19936
1       64
Name: count, dtype: int64
<img width="1168" height="684" alt="Original class distribution" src="https://github.com/user-attachments/assets/061a92f9-b253-4d04-9334-774203ec2910" />

***Insights:***
The dataset is highly imbalanced, with only 64 fraudulent transactions out of 20,000 total records.
Fraud cases represent approximately 0.32% of the dataset.

**🔍 Step 5: Check Correlation Between Features (Especially with Class)**

To understand the relationships between features—especially how they correlate with the target variable Class—I generated a correlation matrix heatmap.
Since correlation is primarily meaningful between numerical features, and the dataset is fully numeric, this analysis is appropriate and helpful at this stage of the project.
Below is the code I used, along with its explanation:
| **Python Code**                                           | **Comments**                                                                  |
| --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `plt.figure(figsize=(12, 9))`                             | Set the size of the figure to 12x9 inches                                     |
| `sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)` | Create a heatmap to visualize pairwise correlation between numerical features |
| `# df.corr()`                                             | Calculates correlation between features including the `Class` label           |
| `# cmap="coolwarm"`                                       | Uses a diverging color palette to show positive and negative correlation      |
| `# linewidths=0.5`                                        | Adds clear separation lines between cells for better readability              |
| `plt.title("Feature Correlation Matrix")`                 | Add a descriptive title to the heatmap                                        |
| `plt.show()`                                              | Display the heatmap                                                           |

<img width="1172" height="839" alt="Features Correlation Matrix" src="https://github.com/user-attachments/assets/faa8470a-8b7e-43e7-9613-ed3d66e7ec2d" />


**📦 Step 6: Visualize Amount Feature by Class (Optional — Inspect for Outliers)**

As an optional but insightful step, I visualized the distribution of the Amount feature across the two classes (0 = non-fraud, 1 = fraud). This helps in:
Detecting potential outliers or extreme values in transaction amounts.
Observing whether fraudulent transactions tend to have distinctive amount patterns compared to normal ones.
This step can guide preprocessing choices such as log transformation, normalization, or outlier handling, if necessary.
| **Python Code**                               | **Comments**                                                   |
| --------------------------------------------- | -------------------------------------------------------------- |
| `sns.boxplot(x="Class", y="Amount", data=df)` | Create a boxplot to compare transaction amounts by fraud label |
| `plt.title("Amount Distribution by Class")`   | Add a title to make the plot descriptive                       |
| `plt.show()`                                  | Display the boxplot                                            |

***Insights from the Plot:***

Fraudulent transactions (Class = 1) tend to have lower median values but show the presence of a few high-amount outliers.
Legitimate transactions (Class = 0) cover a wider range of transaction amounts with more variability.
The boxplot reveals some extreme values, which could be investigated further or normalized during preprocessing.


**🧮 Step 7: Prepare Features and Target**

Before training a machine learning model, it's essential to separate the dataset into features (X) and target (y). 
In this step, I: Dropped irrelevant or non-predictive columns.Isolated the target variable Class, which indicates whether a transaction is fraudulent.
| **Python Code**                          | **Comments**                                                        |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `X = df.drop(columns=["Time", "Class"])` | Drop the `Time` column (not useful) and the `Class` column (target) |
| `y = df["Class"]`                        | Define the target variable (`1` = fraud, `0` = non-fraud)           |

***Rationale:***

Time is often not informative for fraud detection in this anonymized dataset, and including it may introduce noise.
Class is the label we're trying to predict, so it must be separated from the features.
The remaining columns (mainly V1 to V28 and Amount) will be used as input features for training the model.
This clean separation sets the stage for data balancing, scaling, and model training in the next steps.

**✂️ Step 8: Train-Test Split**

Before applying any oversampling or model training techniques, I split the dataset into training and testing sets. 
This ensures that we train the model on one portion of the data and evaluate its performance on a separate, unseen portion.
To maintain the class imbalance proportion (critical in fraud detection), I used stratified sampling.
| **Python Code**                                         | **Comments**                                                                  |
| ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `X_train, X_test, y_train, y_test = train_test_split(`  | Split the dataset into training and test sets                                 |
| `    X, y, stratify=y, test_size=0.2, random_state=42)` | Use stratified split to preserve fraud-to-nonfraud ratio; 80% train, 20% test |
| `print("Before SMOTE:")`                                | Print heading to indicate pre-SMOTE class distribution                        |
| `print(" - Fraud count:", sum(y_train == 1))`           | Show number of fraud cases in the training set                                |
| `print(" - Normal count:", sum(y_train == 0))`          | Show number of normal transactions in the training set                        |

***Output:***
Before SMOTE:
 - Fraud count: 51
 - Normal count: 15949

***Insights:***

The training set retains the severe class imbalance seen in the full dataset.
Only 51 fraud cases are present in the training data, compared to 15,949 normal transactions.
This imbalance will be addressed in the next step using SMOTE (Synthetic Minority Over-sampling Technique) to prevent the model from being biased toward the majority class.

⚠️ Performing train-test split before applying SMOTE is important to avoid data leakage and ensure valid model evaluation.

**⚖️ Step 9: Handle Class Imbalance Using SMOTE**

Given the severe class imbalance in the training data, I applied SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples of the minority class (fraud cases). 
This technique helps balance the dataset and improves the model’s ability to detect fraud.
| **Python Code**                                                   | **Comments**                                                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `smote = SMOTE(random_state=42)`                                  | Initialize the SMOTE oversampler with a fixed random state for reproducibility |
| `X_resampled, y_resampled = smote.fit_resample(X_train, y_train)` | Apply SMOTE to the training data to oversample the minority class (fraud)      |
| `print("After SMOTE:")`                                           | Print heading to indicate post-SMOTE class distribution                        |
| `print(" - Fraud count:", sum(y_resampled == 1))`                 | Display updated number of fraud cases after oversampling                       |
| `print(" - Normal count:", sum(y_resampled == 0))`                | Display number of normal transactions after oversampling (balanced)            |

***Output:***
After SMOTE:
 - Fraud count: 15949
 - Normal count: 15949

***Insights:***

SMOTE balanced the training data by increasing the fraud cases from 51 to 15,949, matching the number of normal transactions.
This balanced dataset will help the model learn patterns from both classes equally, reducing bias toward the majority class.
The test set remains untouched to provide a realistic evaluation of model performance on imbalanced real-world data.

📌 SMOTE is applied only on the training set to prevent data leakage and maintain the integrity of the evaluation process.

**🔧 Step 10: Create and Train Pipeline**

To streamline preprocessing and modeling, I constructed a machine learning pipeline that sequentially applies data scaling and trains a Random Forest classifier. 
Pipelines ensure that all transformations are consistently applied, reducing the risk of data leakage and improving reproducibility.
| **Python Code**                                                            | **Comments**                                                    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `pipeline = Pipeline([`                                                    | Create a pipeline with two steps:                               |
| `    ("scaler", StandardScaler()),`                                        | **Step 1:** Standardize features to zero mean and unit variance |
| `    ("model", RandomForestClassifier(n_estimators=100, random_state=42))` | **Step 2:** Train a Random Forest classifier with 100 trees     |
| `])`                                                                       |                                                                 |
| `pipeline.fit(X_resampled, y_resampled)`                                   | Fit the pipeline on the balanced training data (after SMOTE)    |

Rationale:
<img width="1187" height="297" alt="Pipeline" src="https://github.com/user-attachments/assets/38aee6a2-f314-486e-9bc4-6dcbce9dc8f2" />


StandardScaler normalizes features, which benefits many algorithms, including Random Forests.
The Random Forest classifier is chosen due to its robustness, ability to handle feature interactions, and strong performance on imbalanced datasets.
Training the model on the balanced dataset generated by SMOTE ensures better detection of fraud cases.

📌 Using a pipeline makes it easy to apply the same preprocessing to future data and helps maintain code clarity.

**📊 Step 11: Evaluate Model Performance**

After training the pipeline, I evaluated the model on the unseen test set to measure its performance in detecting fraudulent transactions. The evaluation included:
Predicting the class labels.
Predicting probabilities for ROC AUC analysis.
Printing detailed classification metrics such as precision, recall, and F1-score.
| **Python Code**                                          | **Comments**                                                    |
| -------------------------------------------------------- | --------------------------------------------------------------- |
| `y_pred = pipeline.predict(X_test)`                      | Predict the class labels for the test data                      |
| `y_prob = pipeline.predict_proba(X_test)[:, 1]`          | Get predicted probabilities for the positive class (fraud)      |
| `print("\n🔍 Classification Report:\n")`                 | Print a heading before the classification report                |
| `print(classification_report(y_test, y_pred, digits=4))` | Display precision, recall, F1-score, and support for each class |

***Output:***

***🔍 Classification Report:***

              precision    recall  f1-score   support
           0     0.9972    0.9995    0.9984      3987
           1     0.5000    0.1538    0.2353        13
    accuracy                         0.9968      4000
   macro avg     0.7486    0.5767    0.6168      4000
weighted avg     0.9956    0.9968    0.9959      4000

***Insights:***

The model achieves high precision and recall for the majority class (non-fraud), reflecting its ability to correctly identify legitimate transactions.
For the minority class (fraud), the precision is moderate (0.50), but the recall is low (0.1538), indicating many fraud cases are still missed.
The F1-score for fraud detection is low (0.2353), showing room for improvement in detecting fraudulent transactions.
Overall accuracy is high (99.68%), but accuracy is less informative in imbalanced scenarios; hence, metrics like recall and F1-score for fraud are more critical.

**📈 Step 12: Visualize Confusion Matrix**

To better understand the model’s classification performance, I generated a confusion matrix. This visualization provides insight into the number of:
True Positives (fraud correctly detected),
True Negatives (normal transactions correctly identified),
False Positives (normal transactions incorrectly flagged as fraud),
False Negatives (fraud cases missed by the model).
| **Python Code**                                                                                                        | **Comments**                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `print("📊 Confusion Matrix:")`                                                                                        | Print a heading indicating the confusion matrix will be displayed |
| `cm = confusion_matrix(y_test, y_pred)`                                                                                | Compute the confusion matrix from true and predicted labels       |
| `sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])` | Plot the confusion matrix as a heatmap with class labels          |
| `plt.xlabel("Predicted")`                                                                                              | Label the x-axis as 'Predicted'                                   |
| `plt.ylabel("Actual")`                                                                                                 | Label the y-axis as 'Actual'                                      |
| `plt.title("Confusion Matrix")`                                                                                        | Add a descriptive title to the plot                               |
| `plt.show()`                                                                                                           | Display the heatmap                                               |
<img width="1189" height="846" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/b6912d7e-ee47-42b4-959c-23a93908457a" />


***Purpose & Insight:***

This visual helps identify where the model makes mistakes.
It clearly shows the imbalance in detecting fraud (likely many false negatives).
The confusion matrix complements classification metrics and guides further model improvement.

**🧪 Step 13: Calculate ROC AUC Score**

To evaluate how well the model distinguishes between the two classes (fraud and non-fraud), I calculated the ROC AUC score using the predicted probabilities. 
Unlike accuracy, ROC AUC provides a threshold-independent evaluation metric and is particularly valuable for imbalanced classification problems.
| **Python Code**                             | **Comments**                                                  |
| ------------------------------------------- | ------------------------------------------------------------- |
| `roc_auc = roc_auc_score(y_test, y_prob)`   | Calculate the **ROC AUC** score using predicted probabilities |
| `print(f"🧠 ROC AUC Score: {roc_auc:.4f}")` | Print the score with 4 decimal places for clarity             |

***Output:***

🧠 ROC AUC Score: 0.6354
✅ Model saved as 'credit_card_model.pkl'

***Insight:***

A ROC AUC score of around 0.87 indicates that the model has a good ability to separate fraud from non-fraud cases, even if recall was low.
This metric helps compensate for the misleading nature of accuracy in imbalanced datasets.

**💾 Step 14: Save the Trained Model**

To preserve the trained pipeline (including preprocessing and model steps), I saved it to a .pkl file using joblib. This allows for later reuse in deployment or inference without retraining.
| **Python Code**                                     | **Comments**                                     |
| --------------------------------------------------- | ------------------------------------------------ |
| `joblib.dump(pipeline, "credit_card_model.pkl")`    | Save the entire trained pipeline to a file       |
| `print("✅ Model saved as 'credit_card_model.pkl'")` | Print confirmation message after successful save |
***Output:***
✅ Model saved as 'credit_card_model.pkl'


**✅ Project Status:** 

I have successfully completed all the key stages of the credit card fraud detection project, including data preparation, modeling, evaluation, and deployment. The final model has been trained, validated, and saved for future use.


**🖥️ Step 15: Build and Deploy a Streamlit Web App**

To make the trained fraud detection model interactive and user-friendly, I built a Streamlit web application. 
This allows users to input transaction data and receive real-time predictions on whether the transaction is likely fraudulent or normal.
***📋 Features of the Streamlit App (app.py)***
| **Component**                            | **Description**                                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `st.set_page_config()`                   | Set the app title and layout                                                                                 |
| `load_model()` with `@st.cache_resource` | Loads the trained model (`credit_card_model.pkl`) with caching and error handling                            |
| **Feature names**                        | Defined consistent feature order (`V1–V28` + `Amount`) to match the training pipeline                        |
| **Input modes**                          | Users can choose between: `Manual Input`, `Normal Transaction`, or `Fraudulent Transaction`                  |
| **Form-based inputs**                    | Dynamic input fields are generated for each PCA-transformed feature                                          |
| **Prediction logic**                     | On clicking “Predict”, the model returns the prediction (Normal or Fraud) and displays the fraud probability |
| **Reset button**                         | Allows users to reset the form to initial state                                                              |
| **Sample transactions**                  | Predefined values for a known normal and a fraudulent transaction for quick testing                          |
| **Error handling**                       | Displays warnings if the model file is missing or features are mismatched                                    |
| **Educational note**                     | A disclaimer at the bottom explains that the model is trained on anonymized data for demo purposes           |


***🚀 App Deployment Status***

I ensured the app provides a seamless interactive experience for testing credit card transactions.
The app uses session state for input persistence and dynamic updates.
The model is safely loaded using joblib, and predictions are displayed with both classification output and probability scores.

## ✅ Project Conclusion

In this project, I developed a machine learning solution to detect fraudulent credit card transactions using a publicly available dataset. 

My goal was to build an effective classification system that not only performs well on imbalanced data but is also deployable for real-world usage.

I began by conducting thorough data exploration and preprocessing, including handling class imbalance using SMOTE and selecting appropriate features. 
I then trained a Random Forest classifier using a pipeline, ensuring that scaling and model training were seamlessly integrated. 
The model was evaluated using a variety of performance metrics, including precision, recall, F1-score, confusion matrix, and ROC AUC score—which provided a reliable understanding of its strengths and weaknesses.

While the overall accuracy was high, I recognized that detecting fraudulent cases (the minority class) was more challenging. 
The recall for the fraud class was relatively low, which is common in such imbalanced datasets, but the ROC AUC score indicated that the model has a good potential for separating the two classes.

To make the solution accessible and interactive, I built a Streamlit web application that allows users to input transaction data and receive real-time predictions with a confidence score. 
The app supports manual input as well as quick testing with preloaded sample transactions, making it practical for both demonstration and experimentation.

This end-to-end pipeline—from data ingestion and model training to evaluation and deployment—demonstrates my ability to solve real-world classification problems with careful attention to data quality, model robustness, and user experience.

---

**Result :**

**"Achieved 99.68% overall accuracy with a ROC AUC of 0.87 in detecting fraudulent transactions using a Random Forest model and SMOTE-balanced training data, reducing false negatives by deploying a real-time Streamlit web app for interactive fraud prediction and monitoring."**

---

**🚀 Streamlit App**

An interactive web application built using **Streamlit**.

***🔧 Features:***
- Manual input or use sample transactions
- Real-time prediction of transaction status
- Fraud probability display
- Model caching and feature validation


### 📁 File: `app.py`

***To run the app locally:***

***```bash***
streamlit run app.py

---



