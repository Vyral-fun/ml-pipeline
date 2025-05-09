#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "You will need to enter your API key in the app."
fi

# Run the app with file watcher disabled (avoids PyTorch conflicts)
echo "Starting Yap.market on http://localhost:8501"
streamlit run app.py --server.port=8501 --server.fileWatcherType=none
