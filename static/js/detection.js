document.addEventListener("DOMContentLoaded", function () {
    const fileUpload = document.getElementById("fileUpload");
    const modelSelect = document.getElementById("modelSelect");
    const analyzeButton = document.getElementById("analyzeButton");
    const resultsSection = document.getElementById("results");

    analyzeButton.addEventListener("click", function () {
        const file = fileUpload.files[0];
        const selectedModel = modelSelect.value;

        if (!file) {
            alert("Please upload a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", selectedModel);

        fetch("/upload", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsSection.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                return;
            }

            // Display prediction counts
            const infoDiv = document.createElement("div");
            infoDiv.classList.add("dynamic-info");
            infoDiv.innerHTML = `
  <div class="result-summary">
    <div class="model-info">
      <p><strong>Model:</strong> ${data.model_info.model}</p>
      <p><strong>Feature Selection:</strong> ${data.model_info.feature_selection}</p>
    </div>
    <div class="prediction-counts">
        <p><strong>Predicted Alive:</strong> ${data.prediction_counts.Alive}</p>
        <p><strong>Predicted Dead:</strong> ${data.prediction_counts.Dead}</p>
    </div>
  </div>
`;

resultsSection.appendChild(infoDiv);
                
            document.querySelector("#accuracy").innerText = data.metrics_html.accuracy?.toFixed(4) || "-"; 
            document.querySelector("#logLoss").innerText = data.metrics_html.logLoss?.toFixed(4) || "-";
            document.querySelector("#rocAuc").innerText = data.metrics_html.rocAuc?.toFixed(4) || "-";
            document.querySelector("#precision").innerText = data.metrics_html.precision?.toFixed(4) || "-";
            document.querySelector("#recall").innerText = data.metrics_html.recall?.toFixed(4) || "-";
            document.querySelector("#f1").innerText = data.metrics_html.f1?.toFixed(4) || "-";
            document.querySelector("#sensitivity").innerText = data.metrics_html.sensitivity?.toFixed(4) || "-";
            document.querySelector("#specificity").innerText = data.metrics_html.specificity?.toFixed(4) || "-";
         

           
            
            // Display additional plots
if (data.plots) {
    const plotMap = {
        confusion_matrix: "Confusion Matrix",
        roc_curve: "ROC Curve",
        accuracy_chart: "Accuracy Chart",
        log_loss_chart: "Log Loss Chart",
        smote_distribution: "Class Distribution Before and After SMOTE",
        radar_chart: "Classification Radar Chart",
        sma_convergence: "SMA Convergence Curve",
        pso_convergence: "PSO Convergence Curve"
    };

    for (const [key, label] of Object.entries(plotMap)) {
        if (data.plots[key]) {
            resultsSection.insertAdjacentHTML("beforeend", `
                <h3>${label}</h3>
                <img src="data:image/png;base64,${data.plots[key]}" alt="${label}">
            `);
        }
    }

               
            }
        })
        .catch(error => {
            resultsSection.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        });
    });
});
