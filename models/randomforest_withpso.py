import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,log_loss
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from pyswarms.single import GlobalBestPSO
import matplotlib.pyplot as plt
import numpy as np
import joblib

"""DATA PREPROCESSING"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data_path = "../project/uploads/brca_data_w_subtypes (1).csv" 
data = pd.read_csv(data_path)

X = data.drop(columns=['histological.type', 'HER2.Final.Status', 'ER.Status', 'vital.status', 'PR.Status'], errors='ignore')
y = data[['vital.status']]

X = X.apply(pd.to_numeric, errors='coerce')

X.fillna(X.mean(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Dataset preprocessed successfully.")
print(f"Features shape: {X_scaled.shape}, Target shape: {y.shape}")
print("First few rows of the dataset:")
print(data.head())

zero_count = (y == 0).sum().values[0]
one_count = (y == 1).sum().values[0]
print(f"Count of 0s (Alive): {zero_count}, Count of 1s (Dead): {one_count}")

y.hist(bins=2, grid=False, edgecolor='black')
plt.xlabel("Vital Status (0 = Alive, 1 = Dead)")
plt.ylabel("Count")
plt.title("Distribution of Vital Status")
plt.xticks([0, 1], labels=["Alive", "Dead"])
plt.show()

print("Dataset preprocessed successfully.")
print(f"Features shape: {X_scaled.shape}, Target shape: {y.shape}")
print("First few rows of the dataset:")
print(data.head())

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.ravel()

X_scaled = np.nan_to_num(X_scaled)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"Original dataset shape: {X_scaled.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

original_shape = X_scaled.shape[0]  # Original dataset size before SMOTE
resampled_shape = y_resampled.shape[0]  # Resampled dataset size after SMOTE

labels = ["Original Data", "Resampled Data"]
counts = [original_shape, resampled_shape]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=['blue', 'red'], alpha=0.7, edgecolor='black')
plt.xlabel("Dataset Type")
plt.ylabel("Number of Samples")
plt.title("Original vs Resampled Dataset Size")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib.patches import Patch

y_flat = y.ravel()
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_flat)

orig_alive = np.sum(y_flat == 0)
orig_dead = np.sum(y_flat == 1)
res_alive = np.sum(y_resampled == 0)
res_dead = np.sum(y_resampled == 1)

print("Class Distribution:")
print(f"Before SMOTE - Alive (0): {orig_alive}")
print(f"Before SMOTE - Dead (1):  {orig_dead}")
print(f"After SMOTE  - Alive (0): {res_alive}")
print(f"After SMOTE  - Dead (1):  {res_dead}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].bar(["Alive (0)", "Dead (1)"], [orig_alive, orig_dead], color="skyblue", edgecolor="black")
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Number of Samples")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].bar(["Alive (0)", "Dead (1)"], [res_alive, res_dead], color="salmon", edgecolor="black")
axes[1].set_title("After SMOTE")
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

legend_handles = [
    Patch(facecolor='skyblue', edgecolor='black', label='Before SMOTE'),
    Patch(facecolor='salmon', edgecolor='black', label='After SMOTE')
]

fig.suptitle("Class Distribution Before and After SMOTE", fontsize=14)
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=False, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

"""Particle Swarm Optimization (PSO) for Feature Selection using pyswarms"""

def objective_function(particles):
    fitness = []
    for particle in particles:
        selected_indices = (particle > 0.5).astype(int)
        if np.sum(selected_indices) == 0:
            fitness.append(1e6)
            continue

        selected_features = np.where(selected_indices == 1)[0]
        X_sel = X_resampled[:, selected_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
        )

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        acc = rf_model.score(X_test, y_test)

        penalty = 0.1 * (len(selected_features) / X_resampled.shape[1])
        fitness.append(-(acc - penalty))  # negative accuracy to minimize
    return np.array(fitness)

# --- PSO Setup ---
n_features = X_resampled.shape[1]
bounds = (np.zeros(n_features), np.ones(n_features))
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.5}  # encourage exploration

pso = GlobalBestPSO(n_particles=20, dimensions=n_features, options=options, bounds=bounds)

# --- Run PSO Optimization ---
best_cost, best_position = pso.optimize(objective_function, iters=50)

# --- Get selected features ---
selected_features_pso = np.where(best_position > 0.5)[0]
X_selected = X_resampled[:, selected_features_pso]

joblib.dump(selected_features_pso, "selected_features_pso.pkl")
print("✅ Selected feature indices saved as selected_features_pso.pkl")

# --- Plot PSO Convergence Curve ---
acc_history = [-cost for cost in pso.cost_history]
smooth_acc = pd.Series(acc_history).rolling(window=5).mean()

plt.figure(figsize=(10, 6))
plt.plot(acc_history, label='Raw Accuracy', color='steelblue', marker='o', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('✨ PSO Convergence Curve (RandomForest Classifier)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

selected_feature_names = [X.columns[i] for i in selected_features_pso]

print("\nSelected Feature Names (via PSO):")
print(', '.join([f"{i+1}. {name}" for i, name in enumerate(sorted(selected_feature_names))]))

"""Random Forest for Classification"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ✅ Use all PSO-selected features from full SMOTE-balanced dataset
X_selected = X_resampled[:, selected_features_pso]
y_selected = y_resampled

# ✅ Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100,  max_depth=5, random_state=42)

cv_scores = cross_val_score(rf_model, X_selected, y_selected, cv=kf, scoring='accuracy')
print(f"\n✅ Random Forest Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ✅ Train on full data
rf_model.fit(X_selected, y_selected)

# ✅ Save model
model_filename = "random_forest_model_PSO_features_full_data.pkl"
joblib.dump(rf_model, model_filename)
print(f"📁 Model saved as {model_filename}")

# Save selected feature training data for consistent evaluation
joblib.dump(X_selected, "X_train_selected_pso.pkl")
joblib.dump(y_resampled, "y_train_selected_pso.pkl")
print("✅ Saved PSO training data: X_train_selected_pso.pkl and y_train_selected_pso.pkl")


# ✅ Predict on same full data (for analysis)
y_pred = rf_model.predict(X_selected)

# ✅ Evaluation metrics
accuracy = accuracy_score(y_selected, y_pred)
precision = precision_score(y_selected, y_pred)
recall = recall_score(y_selected, y_pred)
f1 = f1_score(y_selected, y_pred)

# Sensitivity = Recall
sensitivity = recall


tn, fp, fn, tp = confusion_matrix(y_selected, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"\n🎯 Accuracy on Full Dataset (PSO Features): {accuracy:.4f}")
print("\n📈 Evaluation Metrics:")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")

# ✅ Confusion Matrix
cm_rf = confusion_matrix(y_selected, y_pred)

plt.figure(figsize=(6, 5))
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Alive (0)", "Dead (1)"])
disp_rf.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest (PSO Features, Full Data)")
plt.grid(False)
plt.tight_layout()
plt.show()

alive_count = np.sum(y_pred == 0)
dead_count = np.sum(y_pred == 1)

print(f"\n📊 Predicted Alive (0): {alive_count}")
print(f"📊 Predicted Dead (1): {dead_count}")

plt.figure(figsize=(6, 4))
plt.bar(["Predicted Alive (0)", "Predicted Dead (1)"], [alive_count, dead_count],
        color=['blue', 'red'], alpha=0.7, edgecolor='black')
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Prediction Counts (Full Data, PSO Features)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""VISUALIZATION"""

from sklearn.metrics import roc_curve, auc

X_selected = X_resampled[:, selected_features_pso]

X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_selected, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

train_sizes = np.linspace(0.1, 1.0, 20)

train_acc, val_acc = [], []
train_loss, val_loss = [], []

for frac in train_sizes:
    size = int(frac * len(X_train_full))
    X_train = X_train_full[:size]
    y_train = y_train_full[:size]

    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    y_train_prob = rf.predict_proba(X_train)
    y_val_prob = rf.predict_proba(X_val)

    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_val, y_val_pred))

    train_loss.append(log_loss(y_train, y_train_prob))
    val_loss.append(log_loss(y_val, y_val_prob))

final_rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
final_rf.fit(X_train_full, y_train_full)
y_val_prob_full = final_rf.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_prob_full)
roc_auc = auc(fpr, tpr)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Random Forest Evaluation with PSO-Selected Features", fontsize=16, fontweight='bold')

#1. Log Loss
axs[0, 0].plot(train_sizes, train_loss, label="Training Loss", color='steelblue', marker='o')
axs[0, 0].plot(train_sizes, val_loss, label="Validation Loss", color='darkorange', marker='o')
axs[0, 0].set_title("Log Loss")
axs[0, 0].set_xlabel("Training Size Fraction")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Accuracy
axs[0, 1].plot(train_sizes, train_acc, label="Training Accuracy", color='steelblue', marker='o')
axs[0, 1].plot(train_sizes, val_acc, label="Validation Accuracy", color='darkorange', marker='o')
axs[0, 1].set_title("Accuracy")
axs[0, 1].set_xlabel("Training Size Fraction")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3.ROC Curve
axs[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.4f}")
axs[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray')
axs[1, 0].set_title("ROC Curve")
axs[1, 0].set_xlabel("False Positive Rate")
axs[1, 0].set_ylabel("True Positive Rate")
axs[1, 0].legend(loc="lower right")
axs[1, 0].grid(True)

# 4️⃣ AUC Shaded
axs[1, 1].plot(fpr, tpr, color='purple', lw=2, label=f"AUC = {roc_auc:.4f}")
axs[1, 1].fill_between(fpr, 0, tpr, color='purple', alpha=0.2)
axs[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axs[1, 1].set_title("AUC Curve (Shaded)")
axs[1, 1].set_xlabel("False Positive Rate")
axs[1, 1].set_ylabel("True Positive Rate")
axs[1, 1].legend(loc="lower right")
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
)

y_pred = rf_model.predict(X_selected)
y_prob = rf_model.predict_proba(X_selected)[:, 1]

accuracy = accuracy_score(y_resampled, y_pred)
precision = precision_score(y_resampled, y_pred)
recall = recall_score(y_resampled, y_pred)
f1 = f1_score(y_resampled, y_pred)
roc_auc = roc_auc_score(y_resampled, y_prob)
loss = log_loss(y_resampled, y_prob)

print("\n📈 Evaluation Metrics:")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Log Loss      : {loss:.4f}")
print(f"ROC AUC       : {roc_auc:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

import plotly.graph_objects as go
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

categories += [categories[0]]
values += [values[0]]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Model Metrics',
    line=dict(color='royalblue', width=2),
    fillcolor='rgba(65, 105, 225, 0.2)',
    marker=dict(size=6, color='royalblue')
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            tickfont=dict(size=11),
            gridcolor='lightgray',
            gridwidth=1
        ),
        angularaxis=dict(
            tickfont=dict(size=12)
        ),
        bgcolor='white'
    ),
    showlegend=False,
    title=dict(
        text="📊 Model Classification Metrics",
        font=dict(size=16, color='black'),
        x=0.5
    ),
    margin=dict(t=80, b=40, l=50, r=50),
    template='none'
)

fig.show()