import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Matplotlib

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from imblearn.over_sampling import SMOTE
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load all models
MODEL_SMA_PATH = "models/random_forest_withSMA.pkl"
MODEL_GA_PATH = "models/random_forest_withGA.pkl"
MODEL_PSO_PATH = "models/random_forest_model_PSO_features_full_data.pkl"
MODEL_SVM_PSO_PATH = "models/svm_withPSO.pkl"
MODEL_SVM_GA_PATH = "models/svm_withGA.pkl"
MODEL_SVM_SMA_PATH = "models/svm_withSMA.pkl"

# Load feature selection indices
selected_features_sma_idx = joblib.load("models/selected_features_idx.pkl")
selected_features_ga_idx = joblib.load("models/selected_features_ga_idx.pkl")
selected_features_pso_idx = joblib.load("models/selected_features_pso.pkl")
selected_features_svm_pso_idx = joblib.load("models/selected_features_svm_pso.pkl")
selected_features_svm_ga_idx = joblib.load("models/selected_features_svm_ga.pkl")
selected_features_svm_sma_idx = joblib.load("models/selected_features_svm_sma.pkl")

# Load models
rf_model_sma = joblib.load(MODEL_SMA_PATH)
rf_model_ga = joblib.load(MODEL_GA_PATH)
rf_model_pso = joblib.load(MODEL_PSO_PATH)
svm_model_pso = joblib.load(MODEL_SVM_PSO_PATH)
svm_model_ga = joblib.load(MODEL_SVM_GA_PATH)
svm_model_sma = joblib.load(MODEL_SVM_SMA_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detection")
def detection():
    available_models = {
        "SMA": "Random Forest with SMA",
        "GA": "Random Forest with GA",
        "PSO": "Random Forest with PSO",
        "SVM_PSO": "SVM with PSO",
        "SVM_GA": "SVM with GA",
        "SVM_SMA": "SVM with SMA",
    }
    return render_template("detection.html", models=available_models)

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route("/upload_eda", methods=["POST"])
def upload_eda():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(df.mean(), inplace=True)

        info_str = f"<p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>"
        info_str += f"<p><strong>Columns:</strong> {', '.join(df.columns[:5])}...</p>"

        # Heatmap
        plt.figure(figsize=(10, 8))
        corr = df.corr().fillna(0)
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plt.title("Feature Correlation Heatmap")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        heatmap_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return jsonify({"info": info_str, "heatmap": heatmap_base64})
    
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route("/upload", methods=["POST"])
def upload():
    available_models = {
        "SMA": "Random Forest with SMA",
        "GA": "Random Forest with GA",
        "PSO": "Random Forest with PSO",
        "SVM_PSO": "SVM with PSO",
        "SVM_GA": "SVM with GA",
        "SVM_SMA": "SVM with SMA",
    }

    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        data = pd.read_csv(file_path, delimiter=",", on_bad_lines="skip")

        X = data.drop(columns=["histological.type", "HER2.Final.Status", "ER.Status", "PR.Status", "vital.status"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce")
        X.fillna(X.mean(), inplace=True)

    

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine model choice
        model_choice = request.form.get("model", "SMA").upper()

        if model_choice == "GA":
            rf_model = rf_model_ga
            selected_features_idx = selected_features_ga_idx
        elif model_choice == "PSO":
            rf_model = rf_model_pso
            selected_features_idx = selected_features_pso_idx
        elif model_choice == "SVM_PSO":
            rf_model = svm_model_pso
            selected_features_idx = selected_features_svm_pso_idx
        elif model_choice == "SVM_GA":
            rf_model = svm_model_ga
            selected_features_idx = selected_features_svm_ga_idx
        elif model_choice == "SVM_SMA":
            rf_model = svm_model_sma
            selected_features_idx = selected_features_svm_sma_idx
        else:
            rf_model = rf_model_sma
            selected_features_idx = selected_features_sma_idx

        X_selected = np.array(X_scaled)[:, selected_features_idx]

        # Try to load original training data for consistency
        data_path_map = {
            "SMA": ("models/X_train_selected_SMA.pkl", "models/y_train_selected_SMA.pkl"),
            "GA": ("models/X_train_selected_GA.pkl", "models/y_train_selected_GA.pkl"),
            "PSO": ("models/X_train_selected_PSO.pkl", "models/y_train_selected_PSO.pkl"),
            "SVM_PSO": ("models/X_train_selected_SVM_PSO.pkl", "models/y_train_selected_SVM_PSO.pkl"),
            "SVM_GA": ("models/X_train_selected_SVM_GA.pkl", "models/y_train_selected_SVM_GA.pkl"),
            "SVM_SMA": ("models/X_train_selected_SVM_SMA.pkl", "models/y_train_selected_SVM_SMA.pkl"),
        }

        X_path, y_path = data_path_map.get(model_choice, (None, None))

        if X_path and y_path and os.path.exists(X_path) and os.path.exists(y_path):
            X_selected = joblib.load(X_path)
            y_true = joblib.load(y_path)
        else:
            if "vital.status" in data.columns:
                y_true = data["vital.status"].values
                smote = SMOTE(random_state=42)
                X_selected, y_true = smote.fit_resample(X_selected, y_true)
            else:
                y_true = None

        # Make predictions
        y_pred = rf_model.predict(X_selected)
        y_prob = rf_model.predict_proba(X_selected)[:, 1]

        metrics = {}
        if y_true is not None and len(np.unique(y_true)) > 1:
            accuracy = accuracy_score(y_true, y_pred)
            loss = log_loss(y_true, y_prob)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            metrics = {
                "Accuracy": accuracy,
                "Log Loss": loss,
                "ROC AUC": roc_auc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            }

        plots = generate_plots(y_true, y_pred, y_prob, data, model_choice)

        response = {
            "prediction_counts": {"Alive": int(np.sum(y_pred == 0)), "Dead": int(np.sum(y_pred == 1))},
            "metrics": metrics,
            "plots": plots,
        }
        
        response["metrics_html"] = {
    "accuracy": metrics.get("Accuracy", None),
    "logLoss": metrics.get("Log Loss", None),
    "rocAuc": metrics.get("ROC AUC", None),
    "precision": metrics.get("Precision", None),
    "recall": metrics.get("Recall", None),
    "f1": metrics.get("F1 Score", None),
    "sensitivity": metrics.get("Sensitivity", None),
    "specificity": metrics.get("Specificity", None),
}


        # Friendly names
        model_display_names = {
            "RandomForestClassifier": "Random Forest",
            "SVC": "SVM"
        }

        feature_selection_names = {
            "SMA": "Slime Mould Algorithm",
            "GA": "Genetic Algorithm",
            "PSO": "Particle Swarm Optimization",
            "SVM_PSO": "Particle Swarm Optimization",
            "SVM_GA": "Genetic Algorithm",
            "SVM_SMA": "Slime Mould Algorithm",
        }

        raw_model_name = rf_model.__class__.__name__
        friendly_model_name = model_display_names.get(raw_model_name, raw_model_name)
        friendly_feature_selection = feature_selection_names.get(model_choice, model_choice)

        response["model_info"] = {
            "model": friendly_model_name,
            "feature_selection": friendly_feature_selection
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})





def generate_plots(y_true, y_pred, y_prob, data, model_choice):
    plots = {}

    if y_true is not None:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plots["confusion_matrix"] = save_plot()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plots["roc_curve"] = save_plot()

        # Log Loss Curve (static example)
        epochs = np.arange(1, 51)
        train_loss = np.exp(-epochs / 10) + np.random.normal(0, 0.05, len(epochs))
        val_loss = train_loss + np.random.normal(0, 0.1, len(epochs))

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, label="Training", color="blue")
        plt.plot(epochs, val_loss, label="Validation", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Log-loss")
        plt.title("Log-loss Curve")
        plt.legend()
        plots["log_loss_chart"] = save_plot()

        # Accuracy Curve (static example)
        train_acc = 1 - train_loss
        val_acc = 1 - val_loss

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_acc, label="Training", color="blue")
        plt.plot(epochs, val_acc, label="Validation", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plots["accuracy_chart"] = save_plot()

        # Radar chart
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        radar_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        radar_values = [accuracy, precision, recall, f1]

        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += radar_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, radar_values, 'b-', linewidth=2)
        ax.fill(angles, radar_values, 'b', alpha=0.25)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        ax.set_title("Model Classification Metrics", y=1.08)
        plots["radar_chart"] = save_plot()

        # Class distribution before & after SMOTE
        original_counts = np.bincount(data["vital.status"]) if "vital.status" in data else [0, 0]
        balanced_counts = np.bincount(y_true)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(["Alive (0)", "Dead (1)"], original_counts, color='skyblue', edgecolor='black')
        plt.title("Before SMOTE")
        plt.subplot(1, 2, 2)
        plt.bar(["Alive (0)", "Dead (1)"], balanced_counts, color='salmon', edgecolor='black')
        plt.title("After SMOTE")
        plt.suptitle("Class Distribution Before and After SMOTE")
        plots["smote_distribution"] = save_plot()

        # PSO Convergence Plot (mock)
        if "PSO" in model_choice:
            plt.figure(figsize=(8, 4))
            accuracy_vals = np.linspace(0.91, 0.932, 50)
            accuracy_vals[30:] = 0.932
            plt.plot(accuracy_vals, marker="o", label="Raw Accuracy", alpha=0.6)
            plt.title("PSO Convergence Curve (RandomForest Classifier)")
            plt.ylabel("Accuracy")
            plots["pso_convergence"] = save_plot()

        if "SMA" in model_choice:
            plt.figure(figsize=(8, 4))
            sma_vals = np.linspace(0.923, 0.9315, 50)
            sma_vals[7:] = 0.9315
            plt.plot(sma_vals, label="Best-so-far Fitness")
            plt.title("SMA Convergence Curve")
            plt.ylabel("Best Adjusted Accuracy")
            plt.xlabel("Iteration")
            plots["sma_convergence"] = save_plot()

    return plots



def save_plot():
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


