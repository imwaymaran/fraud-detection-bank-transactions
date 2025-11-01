# Fraud Detection

An individual machine learning project focused on detecting fraudulent bank transactions.
The dataset includes **1 million transactions** with severe class imbalance.  
Goal: build a model that improves **recall and precision** beyond a rule-based baseline.

---

## 1. Overview

- Built an end-to-end ML pipeline for fraud detection  
- Conducted data cleaning, EDA, and feature engineering  
- Handled 0.13% minority fraud class  
- Trained and compared Logistic Regression, Random Forest, and XGBoost  
- Evaluated models using Precision, Recall, F1, and ROC-AUC  

**Final model:** XGBoost (no SMOTE, with `scale_pos_weight`)  
**F1 = 0.71**, **Precision = 0.58**, **Recall = 0.92**

---

## 2. Data Description

The dataset contains both categorical and numerical transaction features.

| Feature | Description |
|----------|-------------|
| Type | Type of transaction |
| Amount | Transaction amount |
| NameOrig | Origin account |
| OldBalanceOrg | Balance before transaction (origin) |
| NewBalanceOrig | Balance after transaction (origin) |
| NameDest | Destination account |
| OldBalanceDest | Balance before transaction (destination) |
| NewBalanceDest | Balance after transaction (destination) |
| IsFlaggedFraud | System rule flag (amount > 200,000) |
| IsFraud | True fraud indicator (target) |

---

## 3. Project Structure

```
├── notebooks/  
│   ├── eda.ipynb              # Exploratory Data Analysis  
│   ├── transform.ipynb        # Data cleaning & feature engineering  
│   └── model_train.ipynb      # Model training & evaluation  
├── README.md  
└── .gitignore 
```

---

## 4. Exploratory Data Analysis (EDA)

- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transactions  
- Fraudulent operations **drain origin balances to zero**  
- Numeric fields are **right-skewed** — log transformation applied  
- Fraudulent transactions are much higher in value:
  - **CASH_OUT:** mean \$1.33M vs \$173K  
  - **TRANSFER:** mean \$1.38M vs \$908K  
- Rule-based fraud flag → **Recall = 0.0008**, **Precision = 1.0** → ineffective  

**Insight:** Fraud concentrates in high-value transfers that empty accounts.

---

## 5. Data Transformation

- Dropped: `nameOrig`, `nameDest`, `isFlaggedFraud`  
- Focused on `TRANSFER` and `CASH_OUT` types  
- Handled imbalance (0.13% fraud):
  - Used `class_weight='balanced'`
  - Considered SMOTE (training set only)  
- Engineered new features:
  - `high_risk_type`: flags high-value `TRANSFER`/`CASH_OUT`
  - `orig_diff`, `dest_diff`: measure balance discrepancies

---

## 6. Model Comparison

### With SMOTE

| Model | Precision | Recall | F1 |
|--------|------------|--------|----|
| Logistic Regression | 0.0289 | **0.9897** | 0.0561 |
| Random Forest | **0.5542** | 0.9460 | **0.6990** |
| XGBoost | 0.3375 | **0.9640** | 0.5000 |

### Without SMOTE (`scale_pos_weight`)

| Model | Precision | Recall | F1 |
|--------|------------|--------|----|
| **XGBoost** | 0.5798 | 0.9152 | **0.7099** |

**Decision:**  
XGBoost (no SMOTE) selected for best precision–recall balance and robustness on imbalanced data.

---

## 7. Key Takeaways

- Fraud limited to specific transaction types and high-value amounts  
- Class imbalance strongly impacts model reliability  
- XGBoost effectively identifies frauds without synthetic data  
- Feature interactions (type × amount × balance change) improve detection  

---

## 8. Tech Stack

**Python**, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib, seaborn, Jupyter Notebook

---

## 9. Results Summary

| Metric | Value |
|---------|-------|
| F1 Score | **0.71** |
| Precision | 0.58 |
| Recall | 0.92 |
| Accuracy | 0.998 |
