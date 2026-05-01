# 🌍 Global Pollution Analysis and Energy Recovery

> Classifying countries into pollution severity categories using **Naive Bayes**, **K-Nearest Neighbors**, and **Decision Tree** classifiers — with actionable energy-recovery insights.

---

## 📌 Objective

The goal is to classify countries into pollution severity categories — **Low**, **Medium**, and **High** — based on environmental indicators such as CO₂ emissions, industrial waste, energy consumption, and other pollution-related features.

---

## 🗂️ Project Structure

```
global-pollution-analysis/
│
├── Global_Pollution_Analysis.csv        # Source dataset (add your own)
├── global_pollution_analysis.py         # Main analysis script (mirrors the .ipynb)
├── Global_Pollution_Analysis.ipynb      # Jupyter Notebook (full analysis)
├── requirements.txt                     # Python dependencies
│
├── model_accuracy_comparison.png        # Bar chart: model accuracy
├── confusion_matrices.png               # 3-panel confusion matrices
├── knn_elbow_curve.png                  # KNN optimal-K elbow plot
├── decision_tree_viz.png                # Decision tree visualisation
└── feature_importance.png               # Top-15 feature importances
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/global-pollution-analysis.git
cd global-pollution-analysis

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Place Global_Pollution_Analysis.csv in the project root.

# 5. Run the script
python global_pollution_analysis.py

# — OR open the notebook —
jupyter notebook Global_Pollution_Analysis.ipynb
```

---

## 🔄 Pipeline Overview

### Phase 1 · Data Preprocessing
| Step | Details |
|------|---------|
| **Load** | Read `Global_Pollution_Analysis.csv` into a Pandas DataFrame |
| **Clean** | Median imputation for numeric columns; mode imputation for categorical columns; outlier clipping at 1st–99th percentile |
| **Encode** | `LabelEncoder` applied to `Country` and `Year` columns |
| **Feature Engineering** | `Energy_Per_Capita` = Energy Consumption / Population; `Pollution_Index` = mean of CO₂, waste, and pollution columns |
| **Scale** | `StandardScaler` normalises all numeric features |
| **Split** | 80 / 20 stratified train-test split |

### Phase 2 · Classification
| Model | Strategy | Hyperparameter Tuning |
|-------|----------|-----------------------|
| **Naive Bayes** | GaussianNB for continuous features | — |
| **KNN** | Distance-based classification | Optimal K via 5-fold cross-validation (K = 1 … 20) |
| **Decision Tree** | Recursive binary splitting | GridSearchCV over `max_depth`, `min_samples_split`, `min_samples_leaf` |

Evaluation metrics for all models: **Accuracy · Confusion Matrix · Precision · Recall · F1-score**

### Phase 3 · Reporting & Insights
- Side-by-side accuracy bar chart
- Confusion matrices for all three classifiers
- KNN elbow curve (CV accuracy vs K)
- Decision tree visualisation (depth = 3 for readability)
- Top-15 feature importances from the Decision Tree
- Printed actionable policy recommendations

---

## 📊 Sample Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | — |
| KNN (optimal K) | — |
| Decision Tree | — |

> ℹ️ Results populate after running the notebook/script with your dataset.

---

## 💡 Key Insights & Policy Recommendations

1. **Prioritise emission reduction** in countries flagged with a High Pollution Index.
2. **Invest in energy-efficient infrastructure** to lower energy consumption per capita.
3. **Target industrial waste management** — consistently a top feature driving severity classifications.
4. **Deploy early-warning systems** using model predictions for high-risk regions.
5. **Re-train models annually** as new pollution data becomes available to maintain accuracy.

---

## 📦 Dependencies

| Library | Version |
|---------|---------|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.24 |
| Pandas | ≥ 2.0 |
| Matplotlib | ≥ 3.7 |
| Seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.3 |
| Jupyter | ≥ 1.0 |

---

## 📝 Final Deliverables

- [x] **Jupyter Notebook** (`.ipynb`) — full code, markdown explanations, and inline visualisations
- [x] **Data Visualisations** — saved as `.png` images and embedded in the notebook
- [x] **Final Report** — printed summary of key findings, model evaluations, and actionable recommendations

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

*Built as part of a machine learning analysis on environmental sustainability and pollution-driven energy policy.*
