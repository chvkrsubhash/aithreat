"""
NIDS Configuration Settings
"""

import os

# Server Settings
HOST = '0.0.0.0'
PORT = 14094
DEBUG = True

# Model Settings
MODEL_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'ipca.pkl')

# Multiclass Settings
IS_MULTICLASS = True
LABELS = {
    0: 'BENIGN',
    1: 'DDoS',
    2: 'DoS'
}

# PCA Settings
PCA_COMPONENTS = 35

# Data Settings
DATA_DIR = 'data'
TRAIN_DATA = os.path.join(DATA_DIR, 'CICIDS2017_merged.csv') 
CICIDS_BASE_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

# Detection Settings
INTERFACE = 'Wi-Fi' 
FLOW_TIMEOUT = 60 # seconds
ALERT_THRESHOLD = 0.5 # Lowered for multiclass confidence
LOG_FILE = 'logs/threat_sentry.log'

# Baseline Performance Metrics (from Training)
BASELINE_METRICS = [
    {"Model": "Random Forest", "Accuracy": 0.9980, "Precision": 0.9970, "Recall": 0.9980, "F1": 0.9970},
    {"Model": "XGBoost", "Accuracy": 0.9990, "Precision": 0.9990, "Recall": 0.9990, "F1": 0.9990},
    {"Model": "CNN", "Accuracy": 0.9850, "Precision": 0.9820, "Recall": 0.9850, "F1": 0.9830},
    {"Model": "BiLSTM", "Accuracy": 0.9920, "Precision": 0.9900, "Recall": 0.9920, "F1": 0.9910}
]
