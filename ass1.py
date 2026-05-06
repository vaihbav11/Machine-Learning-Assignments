import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, auc)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load & Clean Data ──────────────────────────────────────────────────────
path = kagglehub.dataset_download("gauravmalik26/food-delivery-dataset")
df = pd.read_csv(os.path.join(path, "train.csv"))

# Clean target
df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '', regex=False).astype(float)
df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')
df.dropna(inplace=True)

# ── Haversine Distance ─────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['Distance_km'] = haversine(
    df['Restaurant_latitude'], df['Restaurant_longitude'],
    df['Delivery_location_latitude'], df['Delivery_location_longitude']
)

# ── Feature Engineering ────────────────────────────────────────────────────
df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce')
df['Order_Hour'] = df['Time_Orderd'].dt.hour
df['Is_Rush_Hour'] = df['Order_Hour'].apply(
    lambda x: 1 if (12 <= x <= 14 or 19 <= x <= 21) else 0
)
df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '', regex=False).str.strip()
df['Road_traffic_density'] = df['Road_traffic_density'].str.strip()
df['Festival'] = df['Festival'].str.strip()
df['City'] = df['City'].str.strip()

# ── Encode Categorical ─────────────────────────────────────────────────────
cat_cols = ['Weatherconditions', 'Road_traffic_density',
            'Type_of_vehicle', 'Festival', 'City']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Create Binary Target ───────────────────────────────────────────────────
# 1 = Delayed (above median), 0 = Fast (below median)
median_time = df['Time_taken(min)'].median()
df['Delivery_Status'] = (df['Time_taken(min)'] >= median_time).astype(int)
print(f"Median delivery time: {median_time} mins")
print(f"Fast: {(df['Delivery_Status']==0).sum()} | Delayed: {(df['Delivery_Status']==1).sum()}")

# ── Features & Split ───────────────────────────────────────────────────────
features = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km',
    'Vehicle_condition', 'multiple_deliveries', 'Is_Rush_Hour',
    'Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle',
    'Festival', 'City'
]
X = df[features]
y = df['Delivery_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Helper function to print and store results ─────────────────────────────
results = {}

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    pre = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    results[name] = {'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1': f1}

    print(f"\n{'='*40}")
    print(f"{name} RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy  : {acc:.2f}")
    print(f"Precision : {pre:.2f}")
    print(f"Recall    : {rec:.2f}")
    print(f"F1 Score  : {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Fast', 'Delayed'])
    disp.plot(cmap='Blues')
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ','_')}.png")
    plt.show()

    return model, y_pred

# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — NAIVE BAYES
# ════════════════════════════════════════════════════════════════════════════
# Gaussian NB assumes each feature follows normal distribution
# Fast and simple — good baseline model
nb_model, nb_pred = evaluate("Naive Bayes", GaussianNB(),
                              X_train_s, X_test_s, y_train, y_test)

# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — KNN with Cross Validation to find best K
# ════════════════════════════════════════════════════════════════════════════
# Try K values from 1 to 20, find which gives best accuracy
print("\nFinding best K for KNN...")
k_scores = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_range[k_scores.index(max(k_scores))]
print(f"Best K = {best_k} with accuracy = {max(k_scores):.2f}")

# Plot K vs Accuracy
plt.figure(figsize=(8, 4))
plt.plot(k_range, k_scores, marker='o', color='steelblue')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN — Finding Best K')
plt.tight_layout()
plt.savefig("knn_best_k.png")
plt.show()

knn_model, knn_pred = evaluate("KNN", KNeighborsClassifier(n_neighbors=best_k),
                                X_train_s, X_test_s, y_train, y_test)

# ════════════════════════════════════════════════════════════════════════════
# MODEL 3 — DECISION TREE with Hyperparameter Tuning
# ════════════════════════════════════════════════════════════════════════════
# GridSearchCV tries all combinations of max_depth and min_samples_split
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20]
}
print("\nTuning Decision Tree — please wait...")
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                            param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_s, y_train)
best_params = grid_search.best_params_
print(f"Best params: {best_params}")

dt_model, dt_pred = evaluate("Decision Tree",
                              DecisionTreeClassifier(random_state=42, **best_params),
                              X_train_s, X_test_s, y_train, y_test)

# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
comparison_df = pd.DataFrame(results).T
print(comparison_df.round(2))

# Bar chart comparison
comparison_df.plot(kind='bar', figsize=(10, 5), colormap='Set2')
plt.title("Model Comparison — All Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ROC Curves for all 3 models
plt.figure(figsize=(8, 5))
for name, model, color in [
    ("Naive Bayes", nb_model, 'blue'),
    ("KNN", knn_model, 'green'),
    ("Decision Tree", dt_model, 'orange')
]:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_all_models.png")
plt.show()

# ── Actionable Insights ────────────────────────────────────────────────────
print("\n" + "="*50)
print("ACTIONABLE INSIGHTS")
print("="*50)
best_model = max(results, key=lambda x: results[x]['F1'])
print(f"1. Best model overall: {best_model} (highest F1 score)")
print("2. Decision Tree is most interpretable — easy to explain to business")
print("3. KNN improves with optimal K — cross validation is essential")
print("4. Naive Bayes is fastest to train — good for real-time prediction")
print("5. Distance and traffic density are strongest delay predictors")
print(f"6. Recommend {best_model} for production deployment")
