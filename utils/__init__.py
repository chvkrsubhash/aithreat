"""
Utility Functions
"""

import logging
import os
import matplotlib.pyplot as plt
import io
import base64
import config

# Setup Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message):
    logging.info(message)
    print(f"[*] {message}")

def generate_plot_base64(fig):
    """Converts a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
