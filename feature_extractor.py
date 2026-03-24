import numpy as np
import pandas as pd

def get_feature_names():
    """Returns the 78 feature names used in CIC-IDS-2017."""
    return [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
        "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
        "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
        "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    ]

def extract_features_from_flow(packets):
    """
    Extracts 78 features from a list of packets belonging to the same flow.
    Returns a DataFrame with the correct column structure.
    """
    if not packets:
        return None
    
    # Initialize a dict with all 78 features as 0
    feature_names = get_feature_names()
    features = {name: 0.0 for name in feature_names}
    
    # Basic real-time extraction logic
    try:
        first_pkt = packets[0]
        if first_pkt.haslayer('IP'):
            # Some basic placeholders derived from the flow
            features["Destination Port"] = first_pkt['TCP'].dport if first_pkt.haslayer('TCP') else (first_pkt['UDP'].dport if first_pkt.haslayer('UDP') else 0)
            features["Total Fwd Packets"] = len(packets)
            features["Flow Duration"] = float(packets[-1].time - packets[0].time) * 1000000 # microseconds
            
            # Packet lengths
            lengths = [len(p) for p in packets]
            features["Fwd Packet Length Max"] = max(lengths)
            features["Fwd Packet Length Min"] = min(lengths)
            features["Fwd Packet Length Mean"] = np.mean(lengths)
            features["Fwd Packet Length Std"] = np.std(lengths)
            features["Max Packet Length"] = max(lengths)
            features["Min Packet Length"] = min(lengths)
            features["Packet Length Mean"] = np.mean(lengths)
    except Exception as e:
        print(f"Error extracting basic features: {e}")

    # Convert to DataFrame (single row)
    df = pd.DataFrame([features])
    
    # Ensure column order matches the training data
    df = df[feature_names]
    
    return df
