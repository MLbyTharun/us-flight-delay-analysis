# ✈️ Flight Delay Prediction Using Machine Learning (1M+ Records)

## 🔍 Problem Statement
Flight delays cause major operational and customer experience challenges for airlines and airports.  
The goal of this project is to **predict whether a flight will be delayed before departure**, using only *pre-departure information* available at scheduling time.

This project focuses on:
- Handling **large-scale, messy real-world data**
- Preventing **data leakage**
- Building **interpretable and high-performing models**

---

## 📊 Dataset
- **Source:** U.S. Flight Delay & Cancellation Dataset  
- **Size:** ~1.5 million sampled rows (from ~29M total records)  
- **Time Period:** Single year snapshot  
- **Data Nature:** Highly imbalanced, noisy, real-world operational data  

### 🎯 Target Variable
- `DELAYED` (Binary)  
  - `1` → Flight delayed  
  - `0` → On-time flight  

---

## 🧠 Approach & Methodology

### 1️⃣ Data Sampling & Memory Efficiency
- Worked with a **GB-scale raw dataset**
- Efficiently sampled ~1.5M rows to balance realism and compute limits
- Ensured reproducibility during sampling

---

### 2️⃣ Exploratory Data Analysis (EDA)
Key analyses included:
- Delay distribution and class imbalance
- Time-based patterns (hour, day, month)
- Airline- and airport-level delay trends
- Identification of high-cardinality categorical features

---

### 3️⃣ Feature Engineering (Leakage-Aware)
Only **pre-departure features** were used.

#### Engineered Features:
- Departure hour extracted from scheduled time
- Frequency (relative) encoding for:
  - Airlines
  - Origin airports
  - Destination airports
- Removal of all **post-flight columns** to avoid leakage

> ⚠️ Columns like `ARR_DELAY`, `ARR_TIME`, `TAXI_IN`,`etc..` were intentionally excluded.

---

### 4️⃣ Modeling Strategy

| Model | Purpose |
|------|--------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear performance model |
| LightGBM | Outperformed Random Forest

#### 📏 Evaluation 
- **ROC-AUC** (chosen due to class imbalance)

---

## 📈 Results
| Model            | ROC-AUC |
|------------------|--------:|
| Logistic Regression | 0.69 |
| Random Forest    | 0.85 |
| LightGBM         | **0.878** |


✔ Random Forest significantly outperformed the baseline  

✔ LightGBM out-performed Random Forest with better score

✔ High performance achieved **without data leakage**

---

## 🔎 Model Interpretability
Feature importance analysis revealed:
- Departure congestion and time-based features are dominant predictors
- Certain airports and airlines have structurally higher delay risk
- Early morning flights tend to be more reliable

---
## 📏 Evaluation Metric: ROC-AUC

Flight delay prediction is a binary classification problem with a **significantly imbalanced target**, where on-time flights greatly outnumber delayed flights.

### Why not Accuracy?
Accuracy can be misleading in this setting.  
A naive model that predicts *“on time”* for every flight could still achieve high accuracy, while failing to identify delayed flights — making it operationally useless.

### Why ROC-AUC?
ROC-AUC (Receiver Operating Characteristic – Area Under Curve) measures a model’s ability to **distinguish between delayed and non-delayed flights across all possible classification thresholds**.

This makes ROC-AUC well-suited for this problem because:
- It is **threshold-independent**
- It remains **robust under class imbalance**
- It evaluates **ranking quality**, not just final class labels

## Threshold Analysis & Tradeoffs (on Random Forest )

While the Random Forest model achieved a ROC-AUC score of 0.85, we performed **threshold analysis** to make the predictions more actionable in real-world scenarios.

By testing multiple probability thresholds, I observed how **Precision**, **Recall**, and **F1-score** change:

| Threshold | Precision | Recall  | F1-score |
|-----------|-----------|---------|----------|
| 0.3       | 0.23      | 0.95    | 0.37     |
| 0.5       | 0.52      | 0.70    | 0.60     |
| 0.7       | 0.91      | 0.36    | 0.51     |

### Key Insight
- **Threshold = 0.7** was found to be the most reliable for predicting flight delays.
- At this threshold, the model predicts a flight as delayed **only when highly confident**, resulting in:
  - **High Precision (0.91)** → very few false alarms; flights flagged as delayed are almost always actually delayed.
  - **Lower Recall (0.36)** → some real delays may be missed, but this ensures operational reliability.

### Tradeoff Between Precision & Recall (Just for Referencing)
- **Precision** measures how many of the flights predicted as delayed were actually delayed.
- **Recall** measures how many of the actual delayed flights were correctly identified.
- Increasing the threshold improves **Precision** but decreases **Recall**.
- Conversely, lowering the threshold improves **Recall** but reduces **Precision**.
- Selecting a threshold depends on business priorities:  
  - Airlines may prefer **higher Precision** to avoid unnecessary disruption alerts.  
  - Air traffic analysts may prefer **higher Recall** to catch as many delays as possible.

By explicitly selecting a threshold and understanding these tradeoffs, the model becomes **practical for real-world use**, not just a numerical benchmark.


### Business Interpretation (As per my Research)
In real-world airline operations, decisions are based on **relative delay risk**, not a fixed probability cutoff.  
A higher ROC-AUC indicates that the model can consistently assign higher risk scores to flights that are more likely to be delayed, enabling risk-based planning and early interventions.

### Summary
ROC-AUC was selected as the primary evaluation metric because it provides a **reliable, imbalance-aware, and operationally meaningful measure of model performance**, making it more appropriate than accuracy for flight delay prediction.

## 📌 Key Insights
- Flight delays are strongly influenced by **system-level congestion**
- Tree-based models capture non-linear effects missed by linear models

## Feature Selection & Model Simplification

After training the baseline Random Forest model, we analyzed **feature importance** to understand which variables contributed meaningfully to flight delay prediction.

### Approach
- Extracted feature importances from the trained Random Forest model.
- Identified features with **very low importance** (near-zero contribution).
- Removed these low-impact features to reduce model complexity.
- Retrained the model using only the most informative features.

### Results
| Model            | ROC-AUC |
|------------------|--------:|
| Logistic Regression | 0.69 |
| Random Forest    | 0.85 |
| LightGBM         | **0.878** |

### Key Insight
- Removing low-importance features resulted in **minimal performance drop** (0.85 → 0.845).
- This indicates that the model was relying primarily on a **small, strong subset of features**.
- The reduced model is:
  - Easier to interpret  
  - Faster to train  
  - Less prone to noise and overfitting  




---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## 🔮 Future Improvements that I have planned
- Experiment with gradient boosting (XGBoost / LightGBM)
- Deploy as a simple API for real-time prediction

---

## 👋 About Me
Aspiring **Data Scientist / ML Engineer** with hands-on experience in:
- Large-scale data analysis
- Feature engineering
- Applied machine learning

Actively seeking **Data Science / ML internships**.
