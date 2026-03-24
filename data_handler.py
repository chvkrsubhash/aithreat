import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
import config

def load_cicids_data(file_path):
    """Loads CIC-IDS-2017 CSV data and cleans column names."""
    try:
        df = pd.read_csv(file_path)
        # Standardize column names (remove leading spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df, is_training=False):
    """
    Handles INF, NaN, and duplicate columns.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN with 0 as in some parts of the notebook or median
    df.fillna(0, inplace=True) 
    
    if is_training:
        # Remove columns with single unique value
        num_unique = df.nunique()
        cols_to_keep = num_unique[num_unique > 1].index
        # But only for numeric columns (excluding Label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        invalid_cols = [c for c in numeric_cols if c not in cols_to_keep]
        df = df.drop(columns=invalid_cols, errors='ignore')
    
    return df

def preprocess_for_training(df):
    """
    Full pipeline for training: cleaning, scaling, and fit PCA.
    """
    # 1. Clean column names and types
    df = clean_data(df, is_training=True)
    
    # 2. Extract target
    target = None
    if 'Label' in df.columns:
        target_raw = df['Label']
        # Map labels to 0, 1, 2 as per notebook
        # BENIGN -> 0, DDoS -> 1, DoS -> 2
        label_map = {'BENIGN': 0, 'DDoS': 1, 'DoS': 2}
        # Handle cases where Label might be slightly different
        target = target_raw.map(lambda x: label_map.get(x, 0)) # Default to BENIGN
        df = df.drop(columns=['Label'])
    
    # 3. Scale
    features = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 4. PCA
    # Use 35 components as in the notebook
    pca = IncrementalPCA(n_components=config.PCA_COMPONENTS, batch_size=500)
    # Fit in batches to mimic IncrementalPCA usage in notebook if data is large
    # For now, just fit_transform
    pca_data = pca.fit_transform(scaled_data)
    
    # Save objects
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(pca, config.PCA_PATH)
    
    return pca_data, target

def preprocess_for_inference(df):
    """
    Full pipeline for real-time inference using saved scaler and PCA.
    """
    # 1. Clean
    df = clean_data(df, is_training=False)
    
    # 2. Drop Label if present
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
        
    # Check if we have the saved models
    if not os.path.exists(config.SCALER_PATH) or not os.path.exists(config.PCA_PATH):
        raise FileNotFoundError("Scaler or PCA model not found for inference. Please train first.")
        
    scaler = joblib.load(config.SCALER_PATH)
    pca = joblib.load(config.PCA_PATH)
    
    if hasattr(scaler, 'feature_names_in_'):
        for c in scaler.feature_names_in_:
            if c not in df.columns:
                df[c] = 0
        features = df[scaler.feature_names_in_]
    else:
        features = df.select_dtypes(include=[np.number])
    
    scaled_data = scaler.transform(features)
    pca_data = pca.transform(scaled_data)
    
    return pca_data

def encode_labels(labels):
    """Encodes categorical labels into numeric ones."""
    le = LabelEncoder()
    return le.fit_transform(labels), le
