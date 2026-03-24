import config
from feature_extractor import extract_features_from_flow
from alert_system import log_alert
from utils.ml_logic import run_inference
import pandas as pd
import numpy as np

# Global selected model (can be updated from app.py)
SELECTED_MODEL = "Best Model"

# Flow management
flows = {} # Key: (src, dst, sport, dport, proto)

def update_selected_model(model_name):
    global SELECTED_MODEL
    SELECTED_MODEL = model_name

def process_packet(packet):
    """Callback for scapy sniff."""
    if not packet.haslayer('IP'): return
    
    # 5-tuple key
    proto = packet['IP'].proto
    src = packet['IP'].src
    dst = packet['IP'].dst
    sport = 0
    dport = 0
    
    if packet.haslayer('TCP'):
        sport = packet['TCP'].sport
        dport = packet['TCP'].dport
    elif packet.haslayer('UDP'):
        sport = packet['UDP'].sport
        dport = packet['UDP'].dport
    
    flow_key = (src, dst, sport, dport, proto)
    
    # Add to flow
    if flow_key not in flows:
        flows[flow_key] = []
    flows[flow_key].append(packet)
    
    # If flow is "complete" (simplified threshold)
    if len(flows[flow_key]) > 10:
        analyze_flow(flow_key)

def analyze_flow(flow_key):
    """Extracts features and predicts using the selected model."""
    packets = flows.pop(flow_key)
    
    # Get DataFrame with 78 features
    data_df = extract_features_from_flow(packets)
    
    if data_df is not None:
        try:
            label, confidence, probs = run_inference(SELECTED_MODEL, data_df)
            
            # If the label is not BENIGN, log an alert
            if label != "BENIGN" and confidence > config.ALERT_THRESHOLD:
                log_alert(flow_key, f"Attack Detected: {label}", confidence)
            
        except Exception as e:
            print(f"Error in real-time analysis: {e}")
