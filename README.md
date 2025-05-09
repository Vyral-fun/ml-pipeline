# Yap-Market Social Listening - Semantic Analyzer

## Overview

This semantic analyzer calculates the relevance of social media content (tweets) to a campaign brief using a production-quality **hybrid bi-encoder and cross-encoder architecture**.

This two-stage approach combines speed and accuracy:

1.  **Bi-Encoder (Fast Filtering):** Uses OpenAI's embedding models (e.g., `text-embedding-3-small`) for rapid initial scoring of all tweets via cosine similarity. This efficiently identifies a set of promising candidates.
2.  **Cross-Encoder (Accurate Re-ranking):** Employs a `sentence-transformers` cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to perform a more computationally intensive but highly accurate relevance assessment on the top candidates selected by the bi-encoder. This provides the final nuanced relevance scores.

## Features

- **Hybrid Bi-Encoder + Cross-Encoder System** - State-of-the-art for relevance ranking.
- **Fast Initial Filtering** - Bi-encoder efficiently processes large volumes of tweets.
- **Accurate Re-ranking** - Cross-encoder provides nuanced final scores for top candidates.
- **Modern Embedding Models** - Uses OpenAI for bi-encoding and `sentence-transformers` for cross-encoding.
- **Bi-Encoder Embedding Caching** - Stores OpenAI embeddings locally to reduce API calls and costs.
- **Twitter API Integration** - Can use Twitter API to fetch actual tweet content.
- **Progress Reporting & Batch Processing** - Manages large datasets efficiently.
- **Test Mode** - Allows quick validation of the two-stage pipeline with sample tweets.
- **Flexible Command-Line Interface** - Customize models, batch sizes, and top-N for re-ranking.
- **Score Distribution Summary** - Provides insights into the relevance score spread.

## Requirements

- Python 3.7+
- OpenAI API key (for bi-encoder embeddings)
- Twitter API credentials (optional, for fetching live tweet content)

## Setup

1. **Create and Activate Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux. For Windows: venv\\Scripts\\activate
```

2. **Install Dependencies:**

```bash
pip install pandas numpy openai requests python-dotenv sentence-transformers
```

_(The `sentence-transformers` library will download the specified cross-encoder model on its first run if not already cached locally.)_

3. **Set Up API Keys in `.env` File:**

Create a `.env` file in the `yap-market` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For fetching live tweet content
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
```

4. **Prepare Campaign Brief:**

Ensure `campaign_brief.txt` (or a custom path via CLI) contains your campaign details.

## Usage

### Command-Line Options

```bash
python semantic_analyzer.py [options]
```

- `--csv PATH`: Path to the input CSV file of tweets (default: `CoralApp is Coming-campaign-report-07-05-2025.csv`). Expected to have a `yapSubmissionUrls` column.
- `--brief PATH`: Path to the campaign brief text file (default: `campaign_brief.txt`).
- `--output PATH`: Path to save the output CSV with relevance scores (default: `CoralApp-campaign-report-with-relevance.csv`).
- `--batch-size N`: Batch size for processing tweets during the bi-encoder pass (default: 10).
- `--bi-model NAME`: OpenAI model ID for the bi-encoder (default: `text-embedding-3-small`).
- `--cross-model NAME`: `sentence-transformers` model ID for the cross-encoder (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `--top-n-rerank N`: Number of top candidates from the bi-encoder stage to re-rank using the cross-encoder (default: 50). Set to 0 to disable cross-encoder re-ranking.
- `--test`: Run in test mode with a predefined set of sample tweets.

### Examples

1. **Run with default settings:**

```bash
python semantic_analyzer.py
```

2. **Specify input/output and re-rank top 100 candidates:**

```bash
python semantic_analyzer.py --csv my_tweets.csv --output analyzed_results.csv --top-n-rerank 100
```

3. **Run in test mode:**

```bash
python semantic_analyzer.py --test
```

4. **Use a different cross-encoder model:**

```bash
python semantic_analyzer.py --cross-model cross-encoder/ms-marco-TinyBERT-L-2-v2
```

### Output CSV Columns

- `tweet_text`: Text content of the tweet.
- `bi_encoder_score`: Initial relevance score (0-10) from the bi-encoder.
- `relevance_score`: Final relevance score (0-10). For top N tweets, this is from the cross-encoder; for others, it's their bi-encoder score.
- `relevance_context`: Explanation of how the final score was derived (e.g., "Re-ranked by cross-encoder..." or "Scored by bi-encoder...").

## How It Works

1. **Campaign Brief Processing:** The campaign brief text is loaded.
2. **Bi-Encoder Pass (Initial Filtering):**
   - The campaign brief is embedded once using the specified OpenAI model.
   - Each tweet URL from the input CSV is processed to extract its text (using Twitter API if configured, otherwise a placeholder).
   - Each tweet text is embedded using the OpenAI model.
   - Cosine similarity is calculated between the campaign brief embedding and each tweet embedding. This raw similarity is scaled to 0-10 and stored as `bi_encoder_score`.
3. **Candidate Selection:** The script selects the top `N` (e.g., 50, configurable by `--top-n-rerank`) tweets based on their `bi_encoder_score`.
4. **Cross-Encoder Pass (Re-ranking):**
   - If `top_n_rerank > 0` and the cross-encoder model is available, each of the top N tweets is paired with the campaign brief: `(campaign_brief, tweet_text)`.
   - The `sentence-transformers` cross-encoder model processes these pairs directly.
   - The cross-encoder outputs a relevance score for each pair. This score (often a logit) is converted to a 0-10 scale (e.g., via a sigmoid function) and becomes the final `relevance_score` for these top tweets.
5. **Result Aggregation:** Tweets re-ranked by the cross-encoder get their final scores from it. Other tweets (not in the top N for re-ranking) retain their `bi_encoder_score` as their final `relevance_score`.

This hybrid approach ensures that computationally expensive, high-accuracy cross-encoding is only performed on a reduced set of promising candidates, making the system efficient and effective.

## Performance & Model Choices

- **OpenAI API Costs:** The bi-encoder stage involves API calls to OpenAI for embeddings. Caching is implemented to reduce costs on repeated texts.
- **Cross-Encoder Models:** `sentence-transformers` offers various pre-trained cross-encoders. Smaller models (like `MiniLM` or `TinyBERT` versions) are faster but might be slightly less accurate than larger models. The default `cross-encoder/ms-marco-MiniLM-L-6-v2` offers a good balance.
- **Local Computation:** The cross-encoder runs locally (downloading the model on first use). Performance will depend on your CPU/GPU capabilities for this stage.

## Future Enhancements

- Option to use `sentence-transformers` bi-encoder models for a fully local pipeline.
- Batch prediction for the cross-encoder if analyzing a very large `top_n_rerank` set.
- More sophisticated scaling or calibration of cross-encoder scores.
- Integration with vector databases for managing and querying bi-encoder embeddings at scale.

## References

- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Sentence-Transformers Cross-Encoders](https://www.sbert.net/docs/usage/cross-encoder.html)
