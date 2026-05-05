# Machine-Learning-Assignments

This is assignment repository for certification in TubeDude.

>Food Delivery Time Prediction — ML Assignment 1


End-to-end ML project on 45,000+ real delivery records. Phase 1 covers data cleaning, Haversine distance calculation, and rush hour feature engineering. Phase 2 builds a Linear Regression model (MAE: 5.65 mins, R²: 0.44) to predict delivery time and a Logistic Regression classifier (AUC: 0.80) to categorize deliveries as Fast or Delayed. Key findings: traffic density and distance are strongest predictors of delay.


Stack: Python, Pandas, Scikit-learn, Matplotlib


>Global Pollution Analysis and Energy Recovery — ML Assignment 2


End-to-end ML pipeline on global pollution data across 20 countries. EDA includes correlation heatmap and CO2 emissions bar chart. Linear Regression predicts Energy Recovery (MAE: 13.81 GWh, R²: 0.16). Logistic Regression classifies countries into Low/Medium/High pollution severity with 99% accuracy. Key insight: Industrial waste and CO2 emissions are strongest predictors of energy recovery potential.


Stack: Python, Pandas, Scikit-learn, Seaborn, Matplotlib


>Food Delivery Classification — Naive Bayes, KNN & Decision Tree — ML Assignment 3


Binary classification of food deliveries as Fast or Delayed using three ML models. Data preprocessing includes Haversine distance calculation, rush hour feature engineering, and label encoding. KNN optimized using cross-validation to find best K. Decision Tree tuned using GridSearchCV on max_depth and min_samples_split. All three models compared on Accuracy, Precision, Recall, and F1 score with confusion matrices and ROC curves.


Stack: Python, Pandas, Scikit-learn, Matplotlib


>Food Delivery CNN Classification — Deep Learning  ML Assignment 3



Applied Conv1D CNN on tabular food delivery data to classify deliveries as Fast or Delayed. Features engineered include Haversine distance and rush hour flags. CNN architecture uses two Conv1D layers with BatchNormalization and Dropout for regularization. Model validated using 5-Fold Stratified Cross Validation. Hyperparameter tuning performed across filter sizes and learning rates. Compared against Logistic Regression baseline.


Stack: Python, TensorFlow, Keras, scikit-learn, Pandas, Matplotlib































# 🤖 Machine Learning Assignments & Projects
### By Vaibhav Chaturvedi | CSE AI Graduate

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📁 Repository Structure

```
ML-Assignments/
├── Food_Delivery/
│   ├── Assignment_1_Linear_Logistic.py
│   ├── Assignment_3_NaiveBayes_KNN_DT.py
│   ├── Assignment_4_CNN.py
│   └── Assignment_7_CNN_Clustering_Apriori.py
├── Global_Pollution/
│   ├── Assignment_2_Linear_Logistic.py
│   ├── Assignment_5_NaiveBayes_KNN_DT.py
│   └── Assignment_6_CNN_Clustering.py
└── README.md
```

---

## 📊 All 9 ML Assignments — Complete Overview

### 🍕 FOOD DELIVERY TIME PREDICTION (Dataset: Kaggle — Gaurav Malik)

---

#### ✅ Assignment 1 — Linear Regression + Logistic Regression
**Objective:** Predict delivery time (regression) + classify Fast vs Delayed (classification)

| Model | Metric | Score |
|---|---|---|
| Linear Regression | R² | 0.44 |
| Linear Regression | MAE | 5.65 mins |
| Logistic Regression | AUC | 0.80 |
| Logistic Regression | Accuracy | ~73% |

**Key Techniques:**
- Haversine formula for geographic distance calculation
- Rush hour feature engineering
- Label encoding for categorical variables
- StandardScaler normalization

---

#### ✅ Assignment 3 — Naive Bayes + KNN + Decision Tree
**Objective:** Binary classification of deliveries as Fast or Delayed using 3 models

| Model | Accuracy | F1 Score |
|---|---|---|
| Naive Bayes | ~0.70 | ~0.69 |
| KNN (optimal K) | ~0.74 | ~0.73 |
| Decision Tree (tuned) | ~0.76 | ~0.75 |

**Key Techniques:**
- Cross-validation to find optimal K for KNN
- GridSearchCV for Decision Tree hyperparameter tuning (max_depth, min_samples_split)
- ROC curves for all 3 models comparison

---

#### ✅ Assignment 4 — CNN (Convolutional Neural Network)
**Objective:** Apply CNN on tabular data for Fast vs Delayed classification

| Metric | Score |
|---|---|
| CNN Accuracy | ~0.75 |
| 5-Fold CV Mean | ~0.74 |
| AUC | ~0.82 |

**Architecture:**
```
Input (11 features) → Conv1D(64) → BatchNorm → MaxPool → Dropout
→ Conv1D(32) → BatchNorm → Dropout → Flatten → Dense(64) → Output(Sigmoid)
```

**Key Techniques:**
- Reshape tabular data to 3D for Conv1D
- EarlyStopping to prevent overfitting
- 5-Fold Stratified Cross Validation
- Hyperparameter tuning (filter sizes, learning rates)

---

#### ✅ Assignment 7 — CNN + K-Means Clustering + Apriori Algorithm
**Objective:** Combine deep learning with unsupervised clustering and association rule mining

**Key Techniques:**
- CNN for classification (Fast vs Delayed)
- K-Means Clustering to group delivery patterns
- Hierarchical Clustering with dendrogram
- Apriori Algorithm for association rule mining
- Neural Network for energy/time prediction
- Model validation with cross-validation

---

### 🌍 GLOBAL POLLUTION ANALYSIS & ENERGY RECOVERY (Generated Dataset)

---

#### ✅ Assignment 2 — Linear Regression + Logistic Regression
**Objective:** Predict energy recovery (GWh) + classify pollution severity (Low/Medium/High)

| Model | Metric | Score |
|---|---|---|
| Linear Regression | R² | 0.16 |
| Linear Regression | MAE | 13.81 GWh |
| Logistic Regression | Accuracy | 99% |
| Logistic Regression | F1 | 0.99 |

**Key Features:**
- Energy consumption per capita engineering
- Yearly trend feature extraction
- Label encoding for Country and Year
- Correlation heatmap + CO2 bar chart EDA

---

#### ✅ Assignment 5 — Naive Bayes + KNN + Decision Tree
**Objective:** Classify pollution severity using 3 models

| Model | Key Metric |
|---|---|
| Naive Bayes | MultinomialNB on pollution indices |
| KNN | Optimal K via cross-validation |
| Decision Tree | Pruned with max_depth + min_samples_split |

**Key Techniques:**
- Same pipeline as Assignment 3 applied to pollution data
- All 3 confusion matrices
- ROC curves comparison

---

#### ✅ Assignment 6 — CNN + Clustering + Neural Network
**Objective:** CNN classification + K-Means/Hierarchical clustering on pollution data

**Key Techniques:**
- CNN for pollution level prediction
- K-Means to group countries by pollution profile
- Hierarchical Clustering with dendrogram visualization
- Neural Network for energy recovery prediction
- MSE and MAE evaluation

---

#### ✅ Assignment 8 — CNN + Apriori (Pollution)
**Objective:** Association rule mining on pollution data + CNN validation

**Key Techniques:**
- Apriori Algorithm for pollution-energy association rules
- Lift and confidence metrics for rule evaluation
- CNN model validation
- Feature analysis and interpretation

---

#### ✅ Assignment 9 — SVM for Deforestation Analysis
**Objective:** Predict deforestation levels using Support Vector Machine

| Metric | Score |
|---|---|
| MAE | Evaluated |
| MSE | Evaluated |
| R² | Evaluated |

**Key Techniques:**
- SVM with RBF, Polynomial, and Linear kernels
- Hyperparameter tuning (C, gamma) with RandomizedSearchCV
- Feature importance analysis
- Cross-validation for model stability

---

## 🚀 CV Projects Built

### 🎤 Live Voice Translation System
Real-time speech-to-speech translation supporting 10+ languages
```
Voice Input → Whisper ASR → Helsinki-NLP → gTTS → Audio Output
```
**Stack:** Python, OpenAI Whisper, HuggingFace, gTTS, Streamlit

---

### 💬 Sentiment & Intent Analysis Pipeline
Dual NLP classifier for customer feedback routing
- **Sentiment:** Positive / Negative / Neutral — SVM achieved **91% accuracy**
- **Intent:** Complaint / Inquiry / Feedback / Support
- Reduces manual ticket triage by **40%**

**Stack:** Python, TF-IDF, SVM, Logistic Regression, Naive Bayes, Matplotlib

---

### 🚫 Hate Speech Detection
Multi-classifier NLP pipeline with **92% accuracy**

**Stack:** Python, TF-IDF, SVM, Naive Bayes, scikit-learn

---

## 🛠️ Tech Stack

```python
Languages    : Python 3.10
ML Libraries : scikit-learn, TensorFlow, Keras, HuggingFace Transformers
DL Models    : CNN (Conv1D), ANN, BERT, DistilBERT, Whisper
NLP          : TF-IDF, SVM, Naive Bayes, Helsinki-NLP
Clustering   : K-Means, Hierarchical, Apriori
Data         : Pandas, NumPy, Matplotlib, Seaborn
Tools        : Jupyter, VS Code, Git, Streamlit, Kaggle
```

---

## 📈 Key Learning Outcomes

- End-to-end ML pipeline: Data cleaning → Feature engineering → Modeling → Evaluation
- Applied **7 different model families** across regression and classification tasks
- Hands-on experience with **deep learning** (CNN, ANN) and **classical ML** (SVM, KNN, NB, DT)
- **Unsupervised learning** — K-Means, Hierarchical Clustering, Apriori
- **NLP pipelines** — TF-IDF, BERT, Transformers
- **Model validation** — Cross-validation, GridSearch, ROC curves, Confusion matrices

---

## 👨‍💻 Author

**Vaibhav Chaturvedi**
📧 vaihbavv11@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/vaibhavv11)
🐙 [GitHub](https://github.com/vaibhavv11)

---

⭐ If this helped you, give it a star!
