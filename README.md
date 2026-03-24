# 🛡️ Network Traffic Classification and Intrusion Detection System

A comprehensive Machine Learning-based Network Intrusion Detection System (NIDS) capable of analyzing both offline datasets and real-time network traffic. The system classifies traffic as normal or malicious using the **CIC-IDS-2017** feature set.

## ✨ Features
- **Offline Analysis**: Train ML models on network datasets (CIC-IDS-2017).
- **Real-time Detection**: Live packet capture and classification using Scapy.
- **ML Models**: Random Forest & XGBoost classifiers with high accuracy.
- **Web Dashboard**: Real-time monitoring interface with live updates and alerts.
- **Alert System**: Automated attack detection notifications.

## 🎯 System Architecture
`Network Traffic → Packet Capture → Feature Extraction → ML Classification → Alert System → Dashboard`

## 📋 Requirements
- Python 3.8+
- Administrator/Root privileges (for packet capture)
- Npcap/WinPcap (on Windows)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: Flask, scapy, pandas, xgboost, scikit-learn, matplotlib)*

### 2. Prepare Dataset
Place your CIC-IDS-2017 CSV files in the `data/` directory. Update `config.py` with the correct filename.

### 3. Train Models
```bash
python train_models.py
```
This will preprocess the data, train the models, and save them to the `models/` directory.

### 4. Run the Dashboard
```bash
python app.py
```
Dashboard available at: `http://localhost:14094`

## 📊 Performance Metrics
Run `python evaluator.py` to generate ROC curves and classification reports in the `results/` folder.

## 📁 Project Structure
- `app.py`: Flask web application
- `config.py`: Configuration settings
- `data_handler.py`: Dataset loading and preprocessing
- `feature_extractor.py`: Feature extraction from packets
- `packet_capture.py`: Real-time packet capture logic
- `intrusion_detector.py`: Real-time detection engine
- `alert_system.py`: Alert management
- `train_models.py`: Model training script
- `models/`: Trained model binaries (.pkl)
- `templates/`: Dashboard HTML
- `static/`: CSS & JS assets

## 🔒 Security Notes
- Requires admin/root privileges for packet capture.
- Dashboard is accessible on `0.0.0.0:14094` for remote access.
- For production, implement HTTPS and authentication.

---
## 🛠️ Troubleshooting

### Import Errors (e.g., "Could not find import of pandas")
If your IDE (like VS Code) shows red squiggly lines for `pandas` or `flask` even after installing dependencies:

1. **Select the Correct Interpreter**:
   - Press `Ctrl + Shift + P` (Windows) or `Cmd + Shift + P` (Mac).
   - Type **"Python: Select Interpreter"**.
   - Select the Python environment where you installed the dependencies (usually the one listed as "Global" or specifically your `venv`).

2. **Run the Diagnostic Script**:
   - Run `python fix_environment.py` in your terminal.
   - This script will check your environment and attempt to fix missing dependencies automatically.

3. **Restart the Language Server**:
   - In VS Code, run "Python: Restart Language Server" from the Command Palette.

---
**Capstone Project - Network Intrusion Detection with XAI**
