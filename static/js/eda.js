document.getElementById("analyzeEDA").addEventListener("click", function () {
    const fileInput = document.getElementById("edaUpload");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);  // This MUST match `request.files["file"]` in Flask

    fetch("/upload_eda", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        // Show the EDA content block
        document.getElementById("eda-content").style.display = "block";

        // Display stats
        document.getElementById("datasetInfo").innerHTML = data.info;

        // Display heatmap image
        const heatmapContainer = document.querySelector(".heatmap-container");
        heatmapContainer.innerHTML = "";  // clear old if any

        const heatmapImg = new Image();
        heatmapImg.src = `data:image/png;base64,${data.heatmap}`;
        heatmapImg.alt = "Feature Correlation Heatmap";
        heatmapImg.style.width = "100%";
        heatmapImg.style.maxWidth = "700px";
        heatmapImg.style.border = "1px solid #ccc";
        heatmapImg.style.borderRadius = "10px";

        heatmapContainer.appendChild(heatmapImg);
    })
    .catch(error => {
        alert("Upload failed: " + error.message);
    });
});
