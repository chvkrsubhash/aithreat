"""
Packet Capture Module
Sniffs network traffic and organizes it into flows.
"""

from scapy.all import sniff
import config
from intrusion_detector import process_packet
import threading

def start_sniffing(interface=config.INTERFACE):
    """Starts the real-time packet capture on the specified interface."""
    print(f"🕵️ Starting real-time sniffing on {interface}...")
    
    # Run sniff in a separate thread or blocking
    sniff(iface=interface, prn=process_packet, store=False)

def run_realtime_threaded():
    """Wrapper to run sniffing in background."""
    thread = threading.Thread(target=start_sniffing, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    # For testing standalone
    start_sniffing()
