#!/usr/bin/env python
"""
Script to run Streamlit with the file watcher disabled to avoid
torch-related errors
"""
import os
import sys
import subprocess

# Set environment variable to disable file watcher
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Path to the Python interpreter in the virtual environment
PYTHON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")

# Path to the app
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

def main():
    # Build the command
    cmd = [PYTHON_PATH, "-m", "streamlit", "run", APP_PATH, "--server.fileWatcherType", "none"]
    
    # Run the command
    process = subprocess.Popen(cmd)
    
    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("\nShutting down...")
        process.terminate()
        process.wait()
        
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
