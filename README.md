# 🧠 Slime Mould Algorithm for Breast Cancer Disease Subtype Classification Using Multiomics Data

This project presents an intelligent web-based tool for classifying **breast cancer subtypes** based on **multiomics data** using advanced optimization algorithms. The core objective is to enhance classification performance through **Slime Mould Algorithm (SMA)**-based **feature selection**, and compare its performance against **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** using **Random Forest** and **SVM classifiers**.

---

## 📁 Project Structure

```
BCD Care/
│
├── app.py                           # Main Flask application
│
├── models/                          # Model scripts and saved .pkl artifacts
│   ├── randomforest_withsma.py
│   ├── randomforest_withga.py
│   ├── randomforest_withpso.py
│   ├── svm_withsma.py
│   ├── svm_withga.py
│   ├── svm_withpso.py
│   ├── *.pkl                        # Trained models and selected feature files
│   └── report.log                   # Logging information
│
├── project/uploads/                # Dataset storage
│   └── brca_data_w_subtypes (1).csv
│
├── static/
│   ├── *.png                        # Logos and image assets
│   ├── css/
│   │   ├── detection.css
│   │   ├── eda.css
│   │   ├── navbar.css
│   │   ├── resources.css
│   │   └── style.css
│   └── js/
│       ├── detection.js
│       └── eda.js
│
├── templates/
│   ├── index.html
│   ├── detection.html
│   ├── eda.html
│   └── resources.html
│
└── uploads/
    └── brca_data_w_subtypes (1).csv  # Duplicate for runtime
```

---

## 🔬 Project Overview

- **Objective:** Classify patients' breast cancer subtypes (Alive/Dead) using multiomics data.
- **Core Technique:** Slime Mould Algorithm (SMA) for biologically-inspired feature selection.
- **Comparative Methods:** Genetic Algorithm (GA), Particle Swarm Optimization (PSO).
- **Classifiers:** Random Forest, SVM (Support Vector Machine).
- **Balancing Method:** SMOTE (Synthetic Minority Oversampling Technique).

---

## 💻 Features

- 🔍 Upload and analyze **multiomics datasets** in `.csv` format.
- 📊 Perform **EDA** with visual feature correlations.
- ⚙️ Select from **six different model pipelines**:
  - Random Forest with SMA, GA, PSO
  - SVM with SMA, GA, PSO
- 📈 Visual performance metrics and comparison:
  - ROC Curve
  - Confusion Matrix
  - Convergence Curves
  - Radar Metrics
  - Class Distribution (SMOTE)

---

## ⚙️ Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/adwityac/bcd-care.git
cd bcd-care
pip install -r requirements.txt
```

### Sample `requirements.txt`

```txt
Flask
Flask-CORS
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
imblearn
lightgbm
deap
pyswarms
```

---

## 🚀 How to Run

```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

---

## 🧪 How It Works

### 1. Data Upload:
Upload your dataset (must contain `vital.status` column).

### 2. Model Selection:
Choose a model from dropdown:
- SMA, GA, PSO (RF or SVM variant)

### 3. Result:
- Prediction (Alive/Dead)
- Metric outputs (Accuracy, F1, AUC, etc.)
- Performance charts
- SMOTE visualization

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision / Recall / F1-Score**
- **ROC AUC**
- **Log Loss**
- **Sensitivity & Specificity**
- **Radar Chart** comparing all

---

## 📈 Visual Outputs

- Feature Correlation Heatmap (EDA)
- Confusion Matrix
- ROC Curve
- Log Loss and Accuracy trends
- Radar Performance Chart
- SMA / PSO Convergence Plot

---

## 🎓 Authors

This project was developed as part of academic research at **Banasthali Vidyapith**:

- Ashmita Mukherjee 
- Adwitya Chakraborty
- Kriti 
- Kajal 


**Supervisor:** Dr. Rahul Kumar Vijay

---

## 📜 License

This project is licensed under the MIT License.
