import time
from utils import log_event

# Store alerts (limited to last 100 for memory)
alerts_history = []

def log_alert(flow_key, attack_type, confidence):
    """Triggered when an attack is detected."""
    src, dst, sport, dport, proto = flow_key
    timestamp = time.strftime("%H:%M:%S")
    message = f"🚨 {attack_type}! {src}:{sport} -> {dst}:{dport} (Conf: {confidence:.2f})"
    log_event(message)
    
    # Store for UI
    alerts_history.append({
        "type": attack_type,
        "src": src,
        "dst": dst,
        "conf": round(confidence, 4),
        "timestamp": timestamp
    })
    
    # Cap history
    if len(alerts_history) > 100:
        alerts_history.pop(0)

def get_recent_alerts():
    return alerts_history[-10:]
