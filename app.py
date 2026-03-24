import os
import time
import threading
import pandas as pd
import io
from flask import Flask, render_template, jsonify, request, send_file
import config

from intrusion_detector import update_selected_model
from utils.ml_logic import run_inference, get_shap_explanations

app = Flask(__name__)

# System Status
status = {
    "selected_model": "Best Model",
    "baseline_metrics": config.BASELINE_METRICS
}

@app.route('/')
def index():
    return render_template('index.html', model_names=["Best Model", "Random Forest", "XGBoost", "CNN", "BiLSTM"])

@app.route('/api/select_model', methods=['POST'])
def select_model():
    model_name = request.json.get('model_name', "Best Model")
    status["selected_model"] = model_name
    update_selected_model(model_name)
    return jsonify({"status": "SUCCESS", "current_model": model_name})

@app.route('/api/analyze_csv', methods=['POST'])
def analyze_csv():
    """Analyzes a single row or chunk of traffic from CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        
        from utils.ml_logic import evaluate_all_models
        metrics = evaluate_all_models(df.copy())
            
        sample_df = df.sample(1)
        
        from utils.ml_logic import get_shap_explanations, get_decision_path, get_counterfactual
        
        label, confidence, probs = run_inference(status["selected_model"], sample_df)
        shap_plots, top_features = get_shap_explanations(status["selected_model"], df)
        
        decision_path = get_decision_path(status["selected_model"], sample_df)
        counterfactual = get_counterfactual(status["selected_model"], sample_df)
        
        # Prepare dataset metadata for audit
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        dataset_meta = {
            "shape": list(df.shape),
            "info": info_str,
            "head": df.head(5).to_dict(orient='records')
        }
        
        # Prepare data snapshot for PDF
        data_snapshot = [{"Feature": f"PC{i+1}", "Value": float(sample_df.values[0, i])} for i in range(min(5, sample_df.shape[1]))]
        
        return jsonify({
            "label": label,
            "confidence": confidence,
            "probs": probs.tolist(),
            "shap_plots": shap_plots,
            "decision_path": decision_path,
            "counterfactual": counterfactual,
            "metrics": metrics,
            "top_features": top_features,
            "data_snapshot": data_snapshot,
            "dataset_meta": dataset_meta
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.json
        from utils.ml_logic import generate_pdf_report
        pdf_content = generate_pdf_report(data)
        
        if pdf_content:
            return send_file(
                io.BytesIO(pdf_content),
                mimetype='application/pdf',
                as_attachment=True,
                download_name='NIDS_Analysis_Report.pdf'
            )
        else:
            return jsonify({"error": "Failed to generate PDF"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/baseline_metrics', methods=['GET'])
def get_baseline_metrics():
    return jsonify(config.BASELINE_METRICS)

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", config.PORT))
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG)
