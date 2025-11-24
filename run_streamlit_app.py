"""
Quick launcher for Streamlit app.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    print("="*70)
    print("Starting Streamlit Web Application...")
    print("="*70)
    print("\nThe app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8502")
    print("\nPress Ctrl+C to stop the server.")
    print("="*70 + "\n")
    
    # Run streamlit on port 8502 (8501 may be in use)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8502"])

