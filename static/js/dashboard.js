/**
 * NIDS Dashboard Logic v2.1
 */

document.addEventListener('DOMContentLoaded', () => {
    const clock = document.getElementById('clock');
    const modelSelect = document.getElementById('model-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const csvFile = document.getElementById('csv-file');
    const shapContainer = document.getElementById('shap-container');
    const analysisResult = document.getElementById('analysis-result');
    const downloadPdfBtn = document.getElementById('download-pdf-btn');

    let isAnalyzing = false;
    let lastAnalysisData = null;

    // Clock
    setInterval(() => {
        clock.textContent = new Date().toLocaleTimeString();
    }, 1000);

    // Model Selection

    modelSelect.addEventListener('change', async () => {
        const modelName = modelSelect.value;
        try {
            await fetch('/api/select_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            });
            console.log(`Switched to model: ${modelName}`);
        } catch (e) {
            console.error("Model select error:", e);
        }
    });

    // CSV Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (isAnalyzing || !csvFile.files[0]) {
            if (!csvFile.files[0]) alert("Please select a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append('file', csvFile.files[0]);

        isAnalyzing = true;
        analyzeBtn.textContent = "Analyzing...";
        analysisResult.innerHTML = "Processing...";
        shapContainer.innerHTML = "<p>Generating global SHAP explanations... this may take a moment.</p>";
        downloadPdfBtn.style.display = 'none';

        try {
            const res = await fetch('/api/analyze_csv', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.error) {
                analysisResult.innerHTML = `<span class="danger">Error: ${data.error}</span>`;
                return;
            }

            // Update Prediction UI
            analysisResult.innerHTML = `
                <div class="result-box">
                    <strong>Prediction:</strong> <span class="accent">${data.label}</span><br>
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
                </div>
            `;

            // Store data for PDF
            lastAnalysisData = data;
            downloadPdfBtn.style.display = 'inline-block';

            // Update SHAP Charts
            if (data.shap_plots) {
                let htmlStr = "";
                if (data.shap_plots.waterfall) htmlStr += `<h4>Local Waterfall</h4><img src="data:image/png;base64,${data.shap_plots.waterfall}" style="width:100%; border-radius:4px; margin-bottom:10px;">`;
                if (data.shap_plots.class_bar) htmlStr += `<h4>Global Class-Wise Matrix</h4><img src="data:image/png;base64,${data.shap_plots.class_bar}" style="width:100%; border-radius:4px; margin-bottom:10px;">`;
                if (data.shap_plots.beeswarm) htmlStr += `<h4>Feature Dependencies Distribution</h4><img src="data:image/png;base64,${data.shap_plots.beeswarm}" style="width:100%; border-radius:4px; margin-bottom:10px;">`;
                shapContainer.innerHTML = htmlStr;
            } else {
                shapContainer.innerHTML = "<p>SHAP plot generation failed or not supported for this model.</p>";
            }

            // Update Logic Boxes
            if (data.decision_path) document.getElementById('decision-path').textContent = data.decision_path;
            if (data.counterfactual) document.getElementById('counterfactual').textContent = data.counterfactual;

            // Update Global Metrics
            if (data.metrics) {
                document.getElementById('metrics-card').style.display = 'flex';
                const tbody = document.getElementById('metrics-body');
                tbody.innerHTML = '';
                data.metrics.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${row.Model}</strong></td>
                        <td class="metric-val prec">${(row.Precision * 100).toFixed(2)}%</td>
                        <td class="metric-val rec">${(row.Recall * 100).toFixed(2)}%</td>
                        <td class="metric-val f1">${(row.F1 * 100).toFixed(2)}%</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            // Update Top 5 Features
            if (data.top_features) {
                const tfBody = document.getElementById('top-features-body');
                tfBody.innerHTML = '';
                data.top_features.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>#${row.Rank}</td>
                        <td>${row.Feature}</td>
                        <td class="metric-val acc">${row.ModelImp ? row.ModelImp.toFixed(4) : 'N/A'}</td>
                        <td class="metric-val prec">${row.SHAP.toFixed(4)}</td>
                    `;
                    tfBody.appendChild(tr);
                });
            }

        } catch (e) {
            console.error("Analysis error:", e);
            analysisResult.innerHTML = '<span class="danger">Analysis failed.</span>';
        } finally {
            isAnalyzing = false;
            analyzeBtn.textContent = "Analyze Sample";
        }
    });

    // PDF Download Handler
    downloadPdfBtn.addEventListener('click', async () => {
        if (!lastAnalysisData) return;
        
        downloadPdfBtn.textContent = "Generating PDF...";
        downloadPdfBtn.disabled = true;

        try {
            const response = await fetch('/api/download_pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastAnalysisData)
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `AI-Threat_Analysis_${new Date().getTime()}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                alert("Failed to generate PDF report.");
            }
        } catch (e) {
            console.error("PDF Download error:", e);
            alert("An error occurred while generating the PDF.");
        } finally {
            downloadPdfBtn.textContent = "Download PDF Report";
            downloadPdfBtn.disabled = false;
        }
    });
});
