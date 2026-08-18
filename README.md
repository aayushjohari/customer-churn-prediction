## 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn, with a focus on identifying potential churners so that customer retention efforts can be targeted effectively.

The project covers the complete Machine Learning workflow, from data cleaning and exploratory data analysis to model evaluation, hyperparameter tuning, model saving, and Streamlit deployment.

---

## Overview

Customer churn — when a customer stops using a company's service — is costly to acquire around, since retaining an existing customer is far cheaper than winning a new one. This project builds a machine learning pipeline that predicts which customers are at risk of churning based on their account, service, and billing information.

---

## 📊 Dataset

- **Dataset:** Telco Customer Churn
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** ~7,000 customers
- **Target Variable:** `Churn`

The dataset contains information about:

- Customer demographics
- Tenure
- Phone and internet services
- Online security and backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges

## Target Variable

```text
Yes → Customer churned
No  → Customer did not churn
```
---

## 🔄 Project Workflow

1. Data Cleaning & EDA

   Fixed blank/incorrectly typed values in TotalCharges

   Explored churn patterns across contract type, tenure, monthly charges, internet service type, and tech support subscription

   Key insight: month-to-month contracts, low tenure, and lack of tech support are strongly associated with higher churn

2. Preprocessing

   Stratified train/test split to preserve churn class balance

   Built a ColumnTransformer pipeline:

    StandardScaler for numerical features

    OneHotEncoder for categorical features

3. Models Evaluated

   Several classification approaches were explored.

     a. Logistic Regression

    -> Logistic Regression was used as the baseline classification model.

     b. Logistic Regression with Class Weighting

    -> class_weight='balanced' was used to give greater importance to the minority churn class.

     c. Random Forest

    -> Random Forest was also trained and evaluated, including a class-weighted version.

     d. Hyperparameter Tuning

    -> GridSearchCV was used to tune Logistic Regression hyperparameters.

 4. 📊 Model Comparison
```
Model	Churn Recall	Churn Precision	ROC-AUC
Logistic Regression (Baseline)	0.56	0.51	0.8422
Logistic Regression (class_weight='balanced')	0.78	0.51	0.8417
Random Forest	0.49	0.62	0.8248
Random Forest (class_weight='balanced')	0.63	0.58	0.8176
Logistic Regression (GridSearchCV)	0.56	0.66	0.8422
```

Recall and precision in this table refer specifically to the churn class (Churn = 1).

### 🏆 Final Model

The final model selected for deployment was:
```
Logistic Regression
        +
class_weight='balanced'
        +
StandardScaler
        +
OneHotEncoder
        +
ColumnTransformer
        +
Pipeline
```
---

## 🧮 Confusion Matrix

The final balanced Logistic Regression model produced the following confusion matrix:
```

                 Predicted
                 0       1
Actual   0      748     287
         1       81     293
```
---

## 💾 Model Saving

The final trained pipeline was saved using Joblib:
```
joblib.dump(
    final_model,
    "models/Customer_churn_model.pkl"
)
```

The saved pipeline contains both the preprocessing steps and the trained Logistic Regression model.

---

## 🌐 Streamlit Application

The trained Machine Learning pipeline was deployed using Streamlit.

The application allows users to enter customer information including:

    Demographics
    Service subscriptions
    Contract information
    Billing information

The application provides:

    Churn / Stay prediction
    Churn probability
    Stay probability
    Visual churn risk indicator

## 📁 Project Structure
```
customer-churn-prediction/
│
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Notebook/
│   └── customer-churn.ipynb
│
├── models/
│   └── Customer_churn_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 🛠️ Tech Stack

Programming : Python

Data Analysis : Pandas , NumPy

Data Visualization : Matplotlib , Seaborn

Machine Learning : Scikit-learn , Logistic Regression , Random Forest , GridSearchCV

Model Deployment : Streamlit

Model Persistence : Joblib

---

🚀 Setup & Installation

1. Clone the repository
```
git clone https://github.com/aayushjohari/customer-churn-prediction.git
```

2. Navigate to the project
```
cd customer-churn-prediction
```

3. Create a virtual environment
```
python -m venv venv

```
4. Activate the environment
```
venv\Scripts\activate
```
  macOS / Linux
```
source venv/bin/activate
```

5. Install dependencies
```
pip install -r requirements.txt
```
6. Run the Streamlit application
```
streamlit run app.py
```
## 🚀 Live Demo

👉 [Try the Customer Churn Prediction App](https://customer-churn-prediction-0401.streamlit.app/)

---

## Future Improvements

- Experiment with additional models and threshold tuning
- Perform deeper feature importance and model interpretability analysis
- Improve the Streamlit UI and add more visual insights
- Monitor model performance with new customer data

---

## Deployment

The final machine learning model is deployed as an interactive Streamlit application.

**Live Application:**  
https://customer-churn-prediction-0401.streamlit.app/