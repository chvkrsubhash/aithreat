import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Dropout
from data_handler import load_cicids_data, preprocess_for_training
import config

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(len(config.LABELS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(100, activation='relu'),
        Dense(len(config.LABELS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    print("🚀 Starting Model Training Logic...")
    
    # Check for data
    if not os.path.exists(config.TRAIN_DATA):
        print(f"❌ Error: Training data not found at {config.TRAIN_DATA}")
        return

    # Load data
    df = load_cicids_data(config.TRAIN_DATA)
    if df is None: return

    # Preprocess
    print("🧹 Preprocessing data (PCA, Scaling)...")
    X, y = preprocess_for_training(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE for class balancing
    print("⚖️ Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 1. Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    joblib.dump(rf, config.RF_MODEL_PATH)
    
    # 2. XGBoost
    print("⚡ Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=len(config.LABELS))
    xgb.fit(X_train_res, y_train_res)
    joblib.dump(xgb, config.XGB_MODEL_PATH)

    # Reshape for DL models (needs 3D input: batch, timesteps, features)
    X_train_dl = X_train_res.reshape((X_train_res.shape[0], X_train_res.shape[1], 1))
    X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 3. CNN
    print("🧬 Training CNN...")
    cnn = build_cnn((X_train_res.shape[1], 1))
    cnn.fit(X_train_dl, y_train_res, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
    cnn.save(os.path.join(config.MODEL_DIR, 'cnn_model.h5'))

    # 4. BiLSTM
    print("🔄 Training BiLSTM...")
    bilstm = build_bilstm((X_train_res.shape[1], 1))
    bilstm.fit(X_train_dl, y_train_res, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
    bilstm.save(os.path.join(config.MODEL_DIR, 'bilstm_model.h5'))

    # Save best model link (let's say XGBoost is the best for now)
    joblib.dump(xgb, config.BEST_MODEL_PATH)

    print(f"✅ Training Complete. Models saved to {config.MODEL_DIR}")

if __name__ == "__main__":
    train()
