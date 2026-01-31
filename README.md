# ✈️ Flight Delay Prediction Using Machine Learning (1M+ Records)

## 🔍 Problem Statement
Flight delays create significant operational challenges and negatively impact passenger experience.  
The objective of this project is to **predict whether a flight will be delayed *before departure*** using only **pre-departure information available at scheduling time**.

This project emphasizes:
- Working with **large-scale, real-world aviation data**
- Strict **data leakage prevention**
- Building **interpretable, production-relevant ML models**

---

## 📊 Dataset
- **Source:** U.S. Flight Delay & Cancellation Dataset  
- **Sample Size:** ~1.5 million rows (from ~29M total records)  
- **Time Period:** Single-year snapshot  
- **Data Characteristics:** Highly imbalanced, noisy, real-world operational data  

### 🎯 Target Variable
- `DELAYED` (Binary)
  - `1` → Flight delayed  
  - `0` → On-time flight  

---

## 🧠 Methodology & Workflow

### 1️⃣ Data Sampling & Memory Efficiency
- Original dataset size exceeded **GB-scale**
- Sampled ~1.5M rows to balance realism and compute constraints
- Sampling strategy ensured **reproducibility and representativeness**

---

### 2️⃣ Exploratory Data Analysis (EDA)
Key analyses included:
- Class imbalance and delay distribution
- Time-based delay patterns (hour, day, month)
- Airline-level and airport-level delay behavior
- Identification of high-cardinality categorical features

---

### 3️⃣ Feature Engineering (Leakage-Aware)
Only **pre-departure features** were used to ensure realistic prediction.

#### Engineered Features
- Departure hour extracted from scheduled departure time
- Frequency (relative) encoding for:
  - Airlines
  - Origin airports
  - Destination airports
- Explicit removal of **post-flight and outcome-related columns**

> ⚠️ Columns such as `ARR_DELAY`, `ARR_TIME`, `TAXI_IN`, etc. were intentionally excluded to prevent data leakage.

---

### 4️⃣ Modeling Strategy

| Model | Purpose |
|------|--------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear ensemble baseline |
| LightGBM | Best-performing gradient boosting model |

#### 📏 Evaluation Metric
- **ROC-AUC**, chosen due to strong class imbalance

---

## 📈 Model Performance

| Model | ROC-AUC |
|------|--------:|
| Logistic Regression | 0.69 |
| Random Forest | 0.85 |
| LightGBM | **0.878** |

**Key Observations**
- Random Forest significantly outperformed the linear baseline  
- LightGBM further improved performance by capturing complex non-linear interactions  
- Strong performance achieved **without any data leakage**

---

## 🔎 Model Interpretability
Feature importance analysis revealed that:
- Time-based congestion features are dominant predictors
- Certain airports and airlines exhibit structurally higher delay risk
- Early-morning departures tend to be more reliable

---

## 📏 Why ROC-AUC?

Flight delay prediction is a **binary classification problem with strong class imbalance**, where on-time flights significantly outnumber delayed ones.

### Why not Accuracy?
Accuracy can be misleading in imbalanced settings.  
A naive model predicting *“on time”* for every flight could still achieve high accuracy while being operationally useless.

### Why ROC-AUC?
ROC-AUC measures a model’s ability to **rank delayed flights higher than non-delayed flights across all possible thresholds**.

It is well-suited here because:
- It is **threshold-independent**
- It remains **robust under class imbalance**
- It evaluates **ranking quality**, not just final class labels

---

## 🎚️ Threshold Analysis & Precision–Recall Tradeoff (Random Forest)

To make predictions actionable, threshold analysis was conducted on the Random Forest model.

| Threshold | Precision | Recall | F1-score |
|---------|-----------|--------|---------|
| 0.3 | 0.23 | 0.95 | 0.37 |
| 0.5 | 0.52 | 0.70 | 0.60 |
| 0.7 | 0.91 | 0.36 | 0.51 |

### Key Insight
- A **0.7 threshold** prioritizes **high-confidence delay predictions**
- High Precision (0.91) → very few false delay alerts  
- Lower Recall (0.36) → some delays may be missed, but alerts are reliable

This threshold was later applied consistently across models (including LightGBM) to enable fair comparison.

### Precision vs Recall (Conceptual Tradeoff)
- **Precision:** How many predicted delays were actually delayed
- **Recall:** How many actual delays were correctly identified
- Higher thresholds → higher precision, lower recall
- Threshold choice depends on business priorities(as per my research):
  - Airlines may prefer **high precision** to avoid unnecessary operational disruptions
  - Analysts may prefer **higher recall** to identify as many delays as possible 

---

## 🧪 Feature Selection & Model Simplification

Feature importance from the Random Forest model was used to evaluate model robustness.

### Approach
- Identified features with near-zero importance
- Removed low-impact features
- Retrained the model using only informative features

### Outcome
- ROC-AUC dropped marginally from **0.85 → 0.845**
- Indicates the model relies on a **compact, strong subset of features**
- Reduced model is:
  - Easier to interpret
  - Faster to train
  - Less sensitive to noise

---

## 📌 Key Takeaways
- Flight delays are driven by **system-level congestion effects**
- Tree-based models outperform linear models on this task
- Gradient boosting (LightGBM) delivered the best overall performance
- Threshold tuning enables business-aligned decision making

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- Jupyter Notebook
- Git & GitHub

---

## 🔮 Future Work
- Incorporate external data (e.g., weather conditions)
- Explore cost-sensitive learning for recall-focused optimization
- Deploy the model as a lightweight inference API

---

## 👋 About Me
Aspiring **Data Scientist / ML Engineer** with hands-on experience in:
- Large-scale data analysis
- Feature engineering
- Applied machine learning

Actively seeking **Data Science / Machine Learning Opportunities**.
