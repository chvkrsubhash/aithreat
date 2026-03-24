import joblib
import config
import os

try:
    scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler.pkl"))
    print("Scaler loaded successfully.")
    print("Expected features:", scaler.n_features_in_)
except Exception as e:
    print("Error loading scaler:", e)
    
try:
    pca = joblib.load(os.path.join(config.MODEL_DIR, "ipca.pkl"))
    print("PCA loaded successfully.")
    print("Expected PCA features:", pca.n_features_in_)
except Exception as e:
    print("Error loading PCA:", e)
