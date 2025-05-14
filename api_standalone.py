#!/usr/bin/env python3
"""
Standalone FastAPI server for tweet analysis with OpenAI embeddings.
This server receives tweets from the scraper and analyzes them against campaign briefs.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Tweet Analysis API",
    description="API for analyzing tweets against campaign briefs using OpenAI embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
API_KEY = os.getenv("INGESTION_API_KEY")
if not API_KEY:
    print("WARNING: INGESTION_API_KEY not set in environment variables")
    API_KEY = "default_insecure_key"  # Only for development

api_key_header = APIKeyHeader(name="X-API-Key")

# Data models
class ScrapedTweet(BaseModel):
    tweet_id: str
    content: str
    author_id: str
    campaign_id: str
    scraped_at: str
    campaign_brief: Optional[str] = None

class TweetAnalysisResponse(BaseModel):
    status: str
    score: float
    text: str
    sentiment: Optional[str] = "neutral"
    context: Optional[str] = None

class SemanticAnalyzer:
    """Analyzer that uses OpenAI embeddings to evaluate tweet relevance"""
    
    def __init__(self, api_key=None):
        """Initialize the semantic analyzer with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-large"
        print(f"Initialized SemanticAnalyzer with model: {self.model}")
    
    def get_embeddings(self, text):
        """Get embeddings for text using OpenAI API"""
        if not text or not text.strip():
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def analyze_tweet(self, campaign_brief, tweet):
        """Analyze tweet relevance to campaign brief using OpenAI embeddings"""
        if not tweet or not campaign_brief:
            print(f"Missing input: campaign_brief empty: {not campaign_brief}, tweet empty: {not tweet}")
            return 0.0, "Empty input"
            
        # Clean input
        campaign_brief = campaign_brief.strip() if isinstance(campaign_brief, str) else ""
        tweet = tweet.strip() if isinstance(tweet, str) else ""
        
        if not tweet or not campaign_brief:
            print(f"Empty after cleaning: campaign_brief empty: {not campaign_brief}, tweet empty: {not tweet}")
            return 0.0, "Empty input after cleaning"
        
        try:
            print(f"\nAnalyzing tweet: {tweet[:50]}...")
            print(f"Using campaign brief: {campaign_brief[:50]}...")
            
            # Get embeddings
            print("Getting campaign brief embedding...")
            campaign_embedding = self.get_embeddings(campaign_brief)
            print("Getting tweet embedding...")
            tweet_embedding = self.get_embeddings(tweet)
            
            if not campaign_embedding or not tweet_embedding:
                print("Failed to get embeddings")
                return 0.0, "Failed to get embeddings"
            
            # Calculate similarity
            print("Calculating similarity...")
            similarity = self.calculate_similarity(campaign_embedding, tweet_embedding)
            print(f"Similarity calculated: {similarity}")
            
            # Scale to 0-10 range
            score = similarity * 10.0
            context = f"Relevance based on: CoralApp features and messaging match: {similarity:.3f}"
            
            print(f"Final score: {score:.2f}/10")
            return score, context
        except Exception as e:
            error_msg = f"Error analyzing tweet: {str(e)}"
            print(error_msg)
            return 0.0, f"Error: {str(e)}"

# Initialize analyzer
analyzer = SemanticAnalyzer()

def get_campaign_brief(campaign_id: str) -> str:
    """Get campaign brief from file or create a default one if not found"""
    # Try multiple locations for the campaign brief
    possible_paths = [
        f"./data/{campaign_id}.json",
        f"./data/{campaign_id.replace('-', '_')}.json",
        "./data/test_campaign.json",
        "./campaign_brief.txt"
    ]
    
    for path in possible_paths:
        try:
            if path.endswith('.json'):
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "campaign_brief" in data:
                        print(f"Loaded campaign brief from {path}")
                        return data["campaign_brief"]
            else:
                with open(path, "r") as f:
                    content = f.read()
                    print(f"Loaded campaign brief from {path}")
                    return content
        except Exception as e:
            print(f"Could not load campaign brief from {path}: {e}")
    
    # If no brief found, create a default one
    print("Using default campaign brief")
    return """
    CoralApp Campaign Brief
    
    CoralApp is a new social media platform designed to connect users with shared interests 
    in a more meaningful way. Our app focuses on creating deeper connections through content 
    curation and interest-based communities.
    
    Key features include interest-based communities, AI-powered content curation, 
    privacy controls, and an intuitive interface.
    
    Target audience: Tech-savvy young adults (18-35) looking for meaningful connections.
    """

def save_analysis_results(tweet, score, context):
    """Save analysis results to file"""
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Prepare the analyzed tweet data
    analyzed_tweet = {
        "tweet_id": tweet.tweet_id,
        "content": tweet.content,
        "author_id": tweet.author_id,
        "score": float(score),
        "context": context,
        "analyzed_at": datetime.now().isoformat()
    }
    
    # Standardize campaign ID format
    campaign_id = tweet.campaign_id
    if "-" in campaign_id and tweet.author_id.lower() in campaign_id.lower():
        # Already using the new format (username-date)
        standardized_id = campaign_id
    else:
        # Using the old format or a custom ID, standardize it
        today = datetime.now().strftime('%Y%m%d')
        standardized_id = f"{tweet.author_id.lower()}-{today}"
        print(f"Standardizing campaign ID from {campaign_id} to {standardized_id}")
    
    # Save to a real-time results file for this campaign
    results_file = f"./data/{standardized_id}-realtime-results.json"
    
    # Load existing results if available
    existing_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []
    
    # Add the new result
    existing_results.append(analyzed_tweet)
    
    # Save the updated results
    with open(results_file, "w") as f:
        json.dump(existing_results, f)
    
    print(f"Saved analysis results to {results_file}")
    return results_file

@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {"status": "ok", "message": "Tweet Analysis API is running"}

@app.post("/ingest-tweet", response_model=TweetAnalysisResponse)
async def ingest_tweet(
    tweet: ScrapedTweet,
    api_key: str = Security(api_key_header)
):
    """Endpoint for receiving scraped tweets from the TypeScript service"""
    # Validate API key
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid API key"
        )
    
    # Get campaign brief - either from the request or from files
    if tweet.campaign_brief:
        print("Using campaign brief from request")
        campaign_brief = tweet.campaign_brief
    else:
        print("Campaign brief not provided in request, loading from files")
        campaign_brief = get_campaign_brief(tweet.campaign_id)
    
    # Process tweet with semantic analyzer
    score, context = analyzer.analyze_tweet(campaign_brief, tweet.content)
    
    # Save analysis results
    results_file = save_analysis_results(tweet, score, context)
    
    print(f"Processed tweet {tweet.tweet_id} with score {score:.2f}/10")
    return {
        "status": "processed", 
        "score": float(score), 
        "text": tweet.content,
        "context": context
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", 8002))
    
    print(f"Starting Tweet Analysis API on port {port}")
    uvicorn.run(
        "api_standalone:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
