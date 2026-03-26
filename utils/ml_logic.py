import joblib
import tensorflow as tf
import os
import io
import base64
from PIL import Image
from fpdf import FPDF
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from . import generate_plot_base64
from data_handler import preprocess_for_inference
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Cache for models and explainers
MODELS = {}
EXPLAINERS = {}

def load_model_from_disk(model_name):
    """Loads the specified model from disk."""
    if model_name in MODELS:
        return MODELS[model_name]

    model_path = ""
    if model_name == "Random Forest":
        model_path = config.RF_MODEL_PATH
    elif model_name == "XGBoost":
        model_path = config.XGB_MODEL_PATH
    elif model_name == "CNN":
        model_path = os.path.join(config.MODEL_DIR, 'cnn_model.h5')
    elif model_name == "BiLSTM":
        model_path = os.path.join(config.MODEL_DIR, 'bilstm_model.h5')
    else:
        model_name = "Best Model"
        model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)
    
    MODELS[model_name] = model
    return model

def run_inference(model_name, data_df):
    """Runs prediction on valid DataFrame data."""
    model = load_model_from_disk(model_name)
    
    # Preprocess
    X = preprocess_for_inference(data_df)
    
    if "CNN" in model_name or "BiLSTM" in model_name:
        X_dl = X.reshape((X.shape[0], X.shape[1], 1))
        probs = model.predict(X_dl)
        pred_idx = np.argmax(probs, axis=1)[0]
        confidence = float(np.max(probs))
    else:
        pred_idx = model.predict(X)[0]
        probs = model.predict_proba(X)
        confidence = float(np.max(probs))
    
    label = config.LABELS.get(pred_idx, "UNKNOWN")
    return label, confidence, probs[0]

def get_shap_explanations(model_name, original_df):
    """Generates a dictionary of SHAP plots (Waterfall, Bar, Dependence)."""
    try:
        model = load_model_from_disk(model_name)
        
        # Create global sample and local sample
        sample_size = min(100, len(original_df))
        df_sample = original_df.sample(sample_size, random_state=42) if len(original_df) > sample_size else original_df
        X_global = preprocess_for_inference(df_sample)
        X_local = X_global[0:1] # For waterfall
        
        plots = {}
        
        if model_name in EXPLAINERS:
            explainer = EXPLAINERS[model_name]
        else:
            explainer = shap.Explainer(model)
            EXPLAINERS[model_name] = explainer
            
        # 1. Local Waterfall Plot
        shap_values_local = explainer(X_local)
        
        if "CNN" in model_name or "BiLSTM" in model_name:
            pred_idx = np.argmax(model.predict(X_local.reshape((X_local.shape[0], X_local.shape[1], 1)), verbose=0), axis=1)[0]
        else:
            pred_idx = model.predict(X_local)[0]
        
        if len(shap_values_local.values.shape) == 3:
            vals_local = shap_values_local.values[0, :, pred_idx]
            base_local = shap_values_local.base_values[0, pred_idx]
        else:
            vals_local = shap_values_local.values[0]
            base_local = shap_values_local.base_values[0]
            
        feature_names = [f"PC{i+1}" for i in range(X_global.shape[1])]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(
            values=vals_local,
            base_values=base_local,
            data=X_local[0],
            feature_names=feature_names
        ), show=False)
        plt.tight_layout()
        plots['waterfall'] = generate_plot_base64(fig)
        plt.close(fig)
        
        # Global Plots
        top_features = []
        if len(X_global) > 1:
            shap_values_global = explainer(X_global)
            
            if len(shap_values_global.values.shape) == 3:
                vals_global = shap_values_global.values[:, :, pred_idx]
            else:
                vals_global = shap_values_global.values
                
            # Correlation Table: Feature | Model Importance | SHAP Importance
            mean_abs_shap = np.mean(np.abs(vals_global), axis=0)
            model_importances = {}
            if hasattr(model, "feature_importances_"):
                m_imp = model.feature_importances_
                for i, val in enumerate(m_imp):
                    model_importances[feature_names[i]] = float(val)
            
            # Combine into top_features list
            top_indices = np.argsort(mean_abs_shap)[::-1][:10] # Top 10 for more detail
            top_features = []
            for i, idx in enumerate(top_indices):
                f_name = feature_names[idx]
                top_features.append({
                    "Rank": int(i+1),
                    "Feature": f_name,
                    "SHAP": float(mean_abs_shap[idx]),
                    "ModelImp": model_importances.get(f_name, 0.0)
                })
                
            # 2. Class-Wise Bar Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            if len(shap_values_global.values.shape) == 3:
                list_of_shap = [shap_values_global.values[:, :, i] for i in range(shap_values_global.values.shape[2])]
                class_names = [config.LABELS.get(i, f"Class {i}") for i in range(len(list_of_shap))]
                shap.summary_plot(list_of_shap, X_global, feature_names=feature_names, class_names=class_names, plot_type="bar", show=False)
            else:
                shap.summary_plot(vals_global, X_global, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plots['class_bar'] = generate_plot_base64(fig)
            plt.close(fig)
            
            # 3. BeeSwarm Distribution Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(vals_global, X_global, feature_names=feature_names, show=False)
            plt.tight_layout()
            plots['beeswarm'] = generate_plot_base64(fig)
            plt.close(fig)
            
            # 4. Dependence Plot
            top_feature_idx = top_indices[0] # Most important feature
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.dependence_plot(top_feature_idx, vals_global, X_global, feature_names=feature_names, show=False)
            plt.tight_layout()
            plots['dependence'] = generate_plot_base64(fig)
            plt.close(fig)

        return plots, top_features
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SHAP error: {e}")
        return None, None

def evaluate_all_models(data_df):
    """Evaluates all models on the provided DataFrame and returns metrics."""
    label_col = None
    for col in data_df.columns:
        if str(col).strip().lower() == 'label':
            label_col = col
            break
            
    if not label_col:
        return None
        
    label_map = {
        'BENIGN': 0, 'BENIGN': 0, 'ATTACK': 1,
        'DDOS': 1, 'DDoS': 1, 'DOS': 2, 'DoS': 2,
        'FTP-BRUTEFORCE': 2, 'SSH-BRUTEFORCE': 2,
        'PORT SCAN': 1, 'INFILTRATION': 1, 'HEARTBLEED': 2,
        'WEB ATTACK': 2, 'BOT': 1
    }
    y_true = data_df[label_col].map(lambda x: label_map.get(str(x).strip().upper(), 0)).values
    
    X = preprocess_for_inference(data_df)
    X_dl = X.reshape((X.shape[0], X.shape[1], 1))
    
    models_to_eval = ["Random Forest", "XGBoost", "CNN", "BiLSTM"]
    metrics = []
    
    for m in models_to_eval:
        try:
            model = load_model_from_disk(m)
            if "CNN" in m or "BiLSTM" in m:
                probs = model.predict(X_dl, verbose=0)
                y_pred = np.argmax(probs, axis=1)
            else:
                y_pred = model.predict(X)
                
            acc = float(accuracy_score(y_true, y_pred))
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics.append({
                "Model": m,
                "Accuracy": acc,
                "Precision": float(p),
                "Recall": float(r),
                "F1": float(f)
            })
        except Exception as e:
            print(f"Error evaluating {m}: {e}")
            
    return metrics

def get_decision_path(model_name, data_df):
    """Extracts tree traverse decision path or surrogate path."""
    try:
        X = preprocess_for_inference(data_df)
        dt_path = os.path.join(config.MODEL_DIR, "decision_tree_model.pkl")
        if os.path.exists(dt_path):
            model = joblib.load(dt_path)
        else:
            model = load_model_from_disk(model_name)
            if "CNN" in model_name or "BiLSTM" in model_name:
                return "Neural Networks do not have a discrete decision path."
            
        if not hasattr(model, "decision_path"):
            return "Model does not support decision path extraction."
            
        node_indicator = model.decision_path(X)
        leaf_id = model.apply(X)
        feature = model.tree_.feature
        threshold = model.tree_.threshold

        path = []
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        for node_id in node_index:
            if leaf_id[0] == node_id:
                val = model.tree_.value[node_id]
                pred_class = config.LABELS.get(np.argmax(val), "UNKNOWN")
                path.append(f"Leaf Node -> Prediction: {pred_class}")
                continue

            if (X[0, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            
            path.append(f"PC{feature[node_id]+1} ({X[0, feature[node_id]]:.2f}) {threshold_sign} {threshold[node_id]:.2f}")
            
        return " ->\n".join(path)
    except Exception as e:
        return f"Could not extract decision path: {e}"

def get_counterfactual(model_name, data_df):
    """Generates a randomized local counterfactual to flip the prediction."""
    try:
        model = load_model_from_disk(model_name)
        X = preprocess_for_inference(data_df)
        is_dl = "CNN" in model_name or "BiLSTM" in model_name
        
        def predict_fn(x_in):
            if is_dl:
                p = model.predict(x_in.reshape((x_in.shape[0], x_in.shape[1], 1)), verbose=0)
                return np.argmax(p, axis=1)[0]
            return model.predict(x_in)[0]

        original_pred = predict_fn(X)
        target_pred = 0 if original_pred != 0 else 1 # Flip to Benign if Malicious, or to Attack (any non-zero) if Benign
        target_name = "BENIGN" if target_pred == 0 else "MALICIOUS (Attack)"
        
        for i in range(1000):
            # Use wider search for Benign to Malicious
            scale = 0.5 if original_pred != 0 else 1.5
            noise = np.random.normal(0, scale, X.shape)
            X_cf = X + noise
            cf_pred = predict_fn(X_cf)
            
            # Check if prediction flipped to the desired target
            is_flipped = (cf_pred == 0) if original_pred != 0 else (cf_pred != 0)
            
            if is_flipped:
                diff = X_cf - X
                top_indices = np.argsort(np.abs(diff[0]))[::-1][:3]
                
                changes = []
                for idx in top_indices:
                    val_diff = diff[0, idx]
                    direction = "Increase" if val_diff > 0 else "Decrease"
                    changes.append(f"- {direction} PC{idx+1} by {abs(val_diff):.2f}")
                    
                return f"To change classification to {target_name}, try:\n" + "\n".join(changes)
                
        return f"No simple counterfactual found to flip to {target_name} within local variance."
    except Exception as e:
        return f"Could not generate counterfactual: {e}"

def generate_pdf_report(data):
    """Generates a PDF bytes object from analysis data."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 10, "AI-Threat Advanced Analysis Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, "Generated by AI-Threat XAI Engine", ln=True, align="C")
        pdf.ln(5)
        
        # Prediction Result Summary (NEW)
        pdf.set_fill_color(240, 248, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 12, f"PREDICTION: {data.get('label', 'UNKNOWN')}  |  CONFIDENCE: {data.get('confidence', 0)*100:.2f}%", ln=True, align="C", border=1, fill=True)
        pdf.ln(5)
        
        # Dataset Overview (NEW)
        if data.get('dataset_meta'):
            meta = data['dataset_meta']
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(220, 53, 69) # Danger/Red for branding
            pdf.cell(0, 10, f"DATASET OVERVIEW: {meta.get('shape', [0,0])[0]} Rows x {meta.get('shape', [0,0])[1]} Columns", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            pdf.set_font("Courier", "", 8)
            # Take only first 1000 chars of info to avoid overflow
            info_txt = meta.get('info', '')[:1000] 
            pdf.multi_cell(0, 4, info_txt, border=1)
            pdf.ln(5)
            
            # Sub-table for Head(5)
            head_data = meta.get('head')
            if head_data and isinstance(head_data, list) and isinstance(head_data[0], dict):
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, "Snapshot (First 5 Rows - Selected Columns):", ln=True)
                pdf.set_font("Helvetica", "", 7)
                # Render only first 6 columns to fit the page
                cols = list(head_data[0].keys())[:6]
                col_width = 190 / max(1, len(cols))
                for c in cols:
                    pdf.cell(col_width, 6, str(c)[:15], border=1)
                pdf.ln()
                for h_row in head_data:
                    if isinstance(h_row, dict):
                        for c in cols:
                            pdf.cell(col_width, 6, str(h_row.get(c, ''))[:20], border=1)
                        pdf.ln()
            pdf.ln(5)
        
        # Model Metrics
        if data.get('metrics'):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "1. Model Performance Metrics", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(50, 8, "Model", border=1, fill=False)
            pdf.cell(40, 8, "Precision", border=1)
            pdf.cell(40, 8, "Recall", border=1)
            pdf.cell(40, 8, "F1-Score", border=1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 10)
            for row in data['metrics']:
                pdf.cell(50, 8, row.get('Model', 'N/A'), border=1)
                pdf.cell(40, 8, f"{row.get('Precision', 0)*100:.2f}%", border=1)
                pdf.cell(40, 8, f"{row.get('Recall', 0)*100:.2f}%", border=1)
                pdf.cell(40, 8, f"{row.get('F1', 0)*100:.2f}%", border=1)
                pdf.ln()
            pdf.ln(5)

        # Top Features
        if data.get('top_features'):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "2. Feature Importance Correlation Matrix", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(20, 8, "Rank", border=1)
            pdf.cell(80, 8, "Feature", border=1)
            pdf.cell(40, 8, "Model Imp.", border=1)
            pdf.cell(40, 8, "SHAP Imp.", border=1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 10)
            for f in data['top_features']:
                pdf.cell(20, 8, f"#{f.get('Rank', '?')}", border=1)
                pdf.cell(80, 8, f.get('Feature', 'N/A'), border=1)
                pdf.cell(40, 8, f"{f.get('ModelImp', 0):.4f}", border=1)
                pdf.cell(40, 8, f"{f.get('SHAP', 0):.4f}", border=1)
                pdf.ln()
            pdf.ln(5)

            pdf.ln(5)

        # Analyzed Traffic Snapshot (NEW)
        if data.get('data_snapshot'):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "3. Analyzed Traffic Snapshot (Top 5 Components)", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(80, 8, "Feature Component", border=1)
            pdf.cell(60, 8, "Value (PCA)", border=1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 10)
            for item in data['data_snapshot']:
                pdf.cell(80, 8, item.get('Feature', 'N/A'), border=1)
                pdf.cell(60, 8, f"{item.get('Value', 0):.6f}", border=1)
                pdf.ln()
            pdf.ln(5)

        # Decision Path & Counterfactual
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, "4. Logic & Counterfactuals", ln=True)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Decision Path:", ln=True)
        pdf.set_font("Courier", "", 9)
        pdf.multi_cell(0, 5, data.get('decision_path', "N/A"))
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Counterfactual Scenario:", ln=True)
        pdf.set_font("Courier", "", 9)
        pdf.multi_cell(0, 5, data.get('counterfactual', "N/A"))
        pdf.ln(10)

        # SHAP Charts
        if data.get('shap_plots'):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "4. Explainable AI Visualizations", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            for name, b64 in data['shap_plots'].items():
                if b64:
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(0, 10, f"Chart: {name.replace('_', ' ').capitalize()}", ln=True)
                    import tempfile
                    import os
                    try:
                        img_data = base64.b64decode(b64)
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp.write(img_data)
                            tmp_path = tmp.name
                        pdf.image(tmp_path, w=190)
                        os.unlink(tmp_path)
                    except Exception as img_e:
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.cell(0, 5, f"Error rendering image: {img_e}", ln=True)
                    pdf.ln(10)

        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        import traceback
        print(f"PDF Gen Error: {e}")
        traceback.print_exc()
        return None
