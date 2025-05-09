# Yap.market Social Listening

## Overview

This Streamlit application calculates the relevance of social media content (tweets) to a campaign brief using OpenAI's embedding models for semantic analysis.

## Features

- **OpenAI Embeddings for Semantic Analysis**: Uses state-of-the-art embeddings to measure content relevance
- **Campaign Analysis**: Analyze individual tweets against a campaign brief
- **Batch Processing**: Process multiple tweets at once from a CSV file
- **Interactive UI**: Visualize relevance scores and get context for the scoring process

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yap-market.git
cd yap-market
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_key_here`

## Running the App

```bash
./run_app.py
```

Or use Streamlit directly:

```bash
streamlit run app.py --server.fileWatcherType=none
```

## Usage

### Campaign Analysis
1. Enter your campaign brief in the text area
2. Add tweets to analyze
3. Click "Analyze Tweets"
4. View the relevance scores and analysis

### Batch Processing
1. Upload a CSV file containing tweets (must have a column with tweet text)
2. Upload a campaign brief text file
3. Select the column containing tweet text
4. Click "Process Batch"
5. View the results table and score distribution

## Deployment

See [deploy_streamlit.md](deploy_streamlit.md) for instructions on deploying to Streamlit Cloud.

## Dependencies

- streamlit
- openai
- pandas
- numpy
- matplotlib
- plotly

## License

[MIT License](LICENSE)
