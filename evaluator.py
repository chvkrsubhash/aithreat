import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import config
from data_handler import load_cicids_data, preprocess_for_training

def evaluate():
    print("📊 Starting Model Evaluation...")
    
    # Load and preprocess data
    df = load_cicids_data(config.TRAIN_DATA)
    if df is None: return

    X, y = preprocess_for_training(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # 1. Evaluate Random Forest
    try:
        print("🔎 Evaluating Random Forest...")
        rf = joblib.load(config.RF_MODEL_PATH)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results.append({"Model": "Random Forest", "Accuracy": acc, "Precision": p, "Recall": r, "F1": f})
    except Exception as e:
        print(f"Error evaluating RF: {e}")

    # 2. Evaluate XGBoost
    try:
        print("🔎 Evaluating XGBoost...")
        xgb = joblib.load(config.XGB_MODEL_PATH)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results.append({"Model": "XGBoost", "Accuracy": acc, "Precision": p, "Recall": r, "F1": f})
    except Exception as e:
        print(f"Error evaluating XGB: {e}")

    # Prepare data for Deep Learning models
    X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 3. Evaluate CNN
    try:
        print("🔎 Evaluating CNN...")
        cnn = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'cnn_model.h5'))
        y_pred_probs = cnn.predict(X_test_dl)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results.append({"Model": "CNN", "Accuracy": acc, "Precision": p, "Recall": r, "F1": f})
    except Exception as e:
        print(f"Error evaluating CNN: {e}")

    # 4. Evaluate BiLSTM
    try:
        print("🔎 Evaluating BiLSTM...")
        bilstm = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'bilstm_model.h5'))
        y_pred_probs = bilstm.predict(X_test_dl)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results.append({"Model": "BiLSTM", "Accuracy": acc, "Precision": p, "Recall": r, "F1": f})
    except Exception as e:
        print(f"Error evaluating BiLSTM: {e}")

    # Save and Print Results
    results_df = pd.DataFrame(results)
    print("\n📈 Model Comparison Table:")
    print(results_df.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\n✅ Evaluation complete. Summary saved to results/model_comparison.csv")

if __name__ == "__main__":
    evaluate()
