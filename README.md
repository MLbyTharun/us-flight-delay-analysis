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

#### 📏 Evaluation Metric
- **ROC-AUC** (chosen due to class imbalance)

---

## 📈 Results

| Model | ROC-AUC |
|------|---------|
| Logistic Regression | ~0.69 |
| Random Forest | **~0.85** |

✔ Random Forest significantly outperformed the baseline  
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

### Business Interpretation
In real-world airline operations, decisions are based on **relative delay risk**, not a fixed probability cutoff.  
A higher ROC-AUC indicates that the model can consistently assign higher risk scores to flights that are more likely to be delayed, enabling risk-based planning and early interventions.

### Summary
ROC-AUC was selected as the primary evaluation metric because it provides a **reliable, imbalance-aware, and operationally meaningful measure of model performance**, making it more appropriate than accuracy for flight delay prediction.

## 📌 Key Insights
- Flight delays are strongly influenced by **system-level congestion**
- Tree-based models capture non-linear effects missed by linear models
- Proper feature engineering is more impactful than trying many models

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
- Incorporate weather data available at scheduling time
- Experiment with gradient boosting (XGBoost / LightGBM)
- Deploy as a simple API for real-time prediction

---

## 👋 About Me
Aspiring **Data Scientist / ML Engineer** with hands-on experience in:
- Large-scale data analysis
- Feature engineering
- Applied machine learning

Actively seeking **Data Science / ML internships**.
