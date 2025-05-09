"""
Fine-tuning module for Yap.market social listening cross-encoder models.
This module handles dataset preparation, model fine-tuning, and evaluation.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import CrossEncoder, InputExample, losses
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openai
import re
import argparse
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CrossEncoderFineTuner:
    """
    Fine-tuning handler for cross-encoder models using Yap.market campaign data.
    Leverages sentence-transformers for fine-tuning pre-trained models on custom data.
    """
    
    def __init__(
        self, 
        base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
        output_dir: str = "fine_tuned_models",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        max_seq_length: int = 256
    ):
        """
        Initialize the fine-tuner with model and training parameters.
        
        Args:
            base_model: The base cross-encoder model to fine-tune
            output_dir: Directory to save fine-tuned models
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps
            max_seq_length: Maximum sequence length for inputs
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize OpenAI client for relevance estimation if needed
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        logger.info(f"Initialized fine-tuner with base model: {base_model}")

    def prepare_dataset_from_csv(
        self, 
        csv_path: str, 
        campaign_brief_path: str,
        label_column: Optional[str] = None,
        content_column: str = "content",
        url_column: Optional[str] = None,
        extract_tweets: bool = False,
        use_openai_scoring: bool = False
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Prepare training dataset from a CSV file and campaign brief.
        
        Args:
            csv_path: Path to the CSV file with tweet data
            campaign_brief_path: Path to campaign brief file
            label_column: Column with relevance scores (if available)
            content_column: Column containing tweet content
            url_column: Column with tweet URLs (if content needs extraction)
            extract_tweets: Whether to extract tweet content from URLs
            use_openai_scoring: Whether to use OpenAI to generate relevance scores
            
        Returns:
            Tuple of training and validation datasets as lists of InputExample
        """
        logger.info(f"Preparing dataset from {csv_path} with brief {campaign_brief_path}")
        
        # Load campaign brief
        with open(campaign_brief_path, 'r', encoding='utf-8') as f:
            campaign_brief = f.read().strip()
            
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        if content_column not in df.columns:
            raise ValueError(f"Content column '{content_column}' not found in CSV")
        
        # Extract or use existing tweet content
        if extract_tweets and url_column and url_column in df.columns:
            logger.info(f"Extracting tweet content from URLs in column: {url_column}")
            tweets = []
            for url in tqdm(df[url_column], desc="Extracting tweets"):
                tweet_text = self._extract_tweet_text(url)
                tweets.append(tweet_text)
            df['extracted_content'] = tweets
            content_column = 'extracted_content'
        
        # Get relevance scores
        if label_column and label_column in df.columns:
            logger.info(f"Using existing labels from column: {label_column}")
            # Normalize scores to 0-1 range if they're not already
            scores = df[label_column].values
            if scores.max() > 1:
                scores = scores / 10.0  # Assuming 0-10 scale
        elif use_openai_scoring:
            logger.info("Generating relevance scores using OpenAI")
            scores = []
            for text in tqdm(df[content_column], desc="Scoring with OpenAI"):
                score = self._estimate_relevance_with_openai(campaign_brief, text)
                scores.append(score)
            df['openai_score'] = scores
            label_column = 'openai_score'
        else:
            raise ValueError("No label column provided and use_openai_scoring is False")
        
        # Create dataset
        train_examples = []
        for i, row in df.iterrows():
            content = row[content_column]
            if not isinstance(content, str) or not content.strip():
                continue
                
            score = row[label_column]
            train_examples.append(InputExample(texts=[campaign_brief, content], label=float(score)))
        
        # Split into training and validation
        train_data, val_data = train_test_split(train_examples, test_size=0.2, random_state=42)
        
        logger.info(f"Created dataset with {len(train_data)} training and {len(val_data)} validation examples")
        return train_data, val_data
    
    def _extract_tweet_text(self, tweet_url: str) -> str:
        """
        Extract tweet text from a URL (placeholder implementation).
        In a production environment, you would use Twitter API.
        """
        # Simple regex extraction of tweet ID
        tweet_id_match = re.search(r'status/(\d+)', tweet_url)
        if not tweet_id_match:
            return f"Could not extract tweet ID from {tweet_url}"
        
        tweet_id = tweet_id_match.group(1)
        return f"Placeholder text for tweet {tweet_id}"
    
    def _estimate_relevance_with_openai(self, campaign_brief: str, tweet_text: str, model: str = "gpt-4") -> float:
        """
        Use OpenAI to estimate relevance score between campaign brief and tweet.
        
        Args:
            campaign_brief: The campaign brief text
            tweet_text: The tweet text
            model: OpenAI model to use
            
        Returns:
            Normalized relevance score (0-1)
        """
        prompt = f"""
        Campaign Brief:
        {campaign_brief}
        
        Tweet:
        {tweet_text}
        
        On a scale of 0 to 1, how relevant is the tweet to the campaign brief?
        Consider:
        - Topical relevance: Does the tweet discuss the same topic?
        - Messaging alignment: Does the tweet convey messages that align with the campaign?
        - Brand mentions: Does the tweet mention relevant brands or products?
        - Call to action: Does the tweet include CTAs relevant to the campaign?
        
        Return only a number between 0 and 1, where 1 is extremely relevant and 0 is completely irrelevant.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            match = re.search(r'(\d+\.?\d*|\.\d+)', score_text)
            if match:
                score = float(match.group(0))
                # Ensure score is between 0 and 1
                return min(1.0, max(0.0, score))
            else:
                logger.warning(f"Could not extract score from OpenAI response: {score_text}")
                return 0.5  # Default to middle score
        except Exception as e:
            logger.error(f"Error estimating relevance with OpenAI: {e}")
            return 0.5
    
    def fine_tune(self, train_data: List[InputExample], val_data: List[InputExample]) -> str:
        """
        Fine-tune a cross-encoder model on the provided dataset.
        
        Args:
            train_data: Training dataset as list of InputExample
            val_data: Validation dataset as list of InputExample
            
        Returns:
            Path to the fine-tuned model
        """
        logger.info(f"Fine-tuning {self.base_model} on {len(train_data)} examples")
        
        # Initialize model
        model = CrossEncoder(self.base_model, num_labels=1, max_length=self.max_seq_length)
        
        # Prepare training dataloader
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        
        # Configure training arguments
        warmup_steps = min(self.warmup_steps, int(len(train_dataloader) * 0.1))
        
        # Train the model
        model.fit(
            train_dataloader=train_dataloader,
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            evaluation_steps=max(10, int(len(train_dataloader) * 0.1)),
            output_path=self.output_dir,
            show_progress_bar=True,
            optimizer_params={'lr': self.learning_rate}
        )
        
        # Evaluate on validation data
        if val_data:
            self.evaluate(model, val_data)
        
        # Save model path
        model_path = os.path.join(self.output_dir, "final")
        model.save(model_path)
        
        logger.info(f"Model fine-tuned and saved to {model_path}")
        return model_path
    
    def evaluate(self, model: CrossEncoder, eval_data: List[InputExample]) -> Dict:
        """
        Evaluate model performance on validation data.
        
        Args:
            model: Trained cross-encoder model
            eval_data: Validation dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(eval_data)} examples")
        
        texts = []
        labels = []
        
        for example in eval_data:
            texts.append(example.texts)
            labels.append(example.label)
        
        # Get predictions
        predictions = model.predict(texts)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        
        logger.info(f"Evaluation results - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(labels, predictions, alpha=0.5)
        plt.xlabel("True Relevance")
        plt.ylabel("Predicted Relevance")
        plt.title("Predicted vs Actual Relevance Scores")
        plt.plot([0, 1], [0, 1], 'r--')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "evaluation_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Evaluation plot saved to {plot_path}")
        
        return {
            "mse": mse,
            "mae": mae,
            "plot_path": plot_path
        }
    
    def predict_relevance(self, model_path: str, campaign_brief: str, tweets: List[str]) -> List[float]:
        """
        Predict relevance scores using a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
            campaign_brief: Campaign brief text
            tweets: List of tweets to score
            
        Returns:
            List of relevance scores (0-1)
        """
        logger.info(f"Predicting relevance with model from {model_path}")
        
        # Load model
        model = CrossEncoder(model_path)
        
        # Prepare pairs
        pairs = [[campaign_brief, tweet] for tweet in tweets]
        
        # Get predictions
        scores = model.predict(pairs)
        
        # Ensure scores are in 0-1 range
        normalized_scores = [min(1.0, max(0.0, float(score))) for score in scores]
        
        return normalized_scores

def main():
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder models for Yap.market")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with tweet data")
    parser.add_argument("--brief", type=str, required=True, help="Path to campaign brief file")
    parser.add_argument("--base-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Base cross-encoder model")
    parser.add_argument("--output-dir", type=str, default="fine_tuned_models", help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI to generate relevance scores")
    parser.add_argument("--label-column", type=str, help="Column with relevance scores")
    parser.add_argument("--content-column", type=str, default="content", help="Column with tweet content")
    
    args = parser.parse_args()
    
    fine_tuner = CrossEncoderFineTuner(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    train_data, val_data = fine_tuner.prepare_dataset_from_csv(
        csv_path=args.csv,
        campaign_brief_path=args.brief,
        label_column=args.label_column,
        content_column=args.content_column,
        use_openai_scoring=args.use_openai
    )
    
    model_path = fine_tuner.fine_tune(train_data, val_data)
    
    logger.info(f"Fine-tuning complete. Model saved to {model_path}")
    
if __name__ == "__main__":
    main()
