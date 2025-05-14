import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
import pandas as pd
import requests
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv
import glob
# No need for sentence_transformers import - using only OpenAI embeddings

# Load environment variables from .env and then api_config.env
load_dotenv() # Loads .env by default
load_dotenv(dotenv_path="api_config.env", override=True) # Loads api_config.env and overrides if keys exist

# Simple version of semantic analyzer that doesn't depend on the full implementation
class SimpleSemanticAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-large"
    
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
            st.error(f"Error getting embedding: {e}")
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
        if not tweet.strip() or not campaign_brief.strip():
            return 0.0, "Empty input"
        
        try:
            # Get embeddings
            campaign_embedding = self.get_embeddings(campaign_brief)
            tweet_embedding = self.get_embeddings(tweet)
            
            # Calculate similarity
            similarity = self.calculate_similarity(campaign_embedding, tweet_embedding)
            
            # Scale to 0-10 range
            score = similarity * 10.0
            context = f"OpenAI similarity: {similarity:.3f}"
            
            return score, context
        except Exception as e:
            st.error(f"Error analyzing tweet: {e}")
            return 0.0, f"Error: {str(e)}"
    
    def analyze_tweets(self, campaign_brief, tweets):
        """Analyze tweets against campaign brief using OpenAI embeddings"""
        # Filter out empty tweets
        valid_tweets = [t for t in tweets if t.strip()]
        if not valid_tweets:
            return pd.DataFrame()
            
        # Analyze each tweet
        results = []
        
        for tweet in valid_tweets:
            # Get similarity score
            score, context = self.analyze_tweet(campaign_brief, tweet)
            
            results.append({
                "Tweet": tweet,
                "Relevance Score": score,
                "Context": context
            })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["Relevance Score"], reverse=True)
            
        return pd.DataFrame(results)

import streamlit as st

st.set_page_config(
    page_title="Yap.market Social Listening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (your original CSS)
# ...

# Format score function
def format_score(score):
    if score >= 7.0:
        return f"<span class='high-score'>{score:.2f}</span>"
    elif score >= 4.0:
        return f"<span class='medium-score'>{score:.2f}</span>"
    else:
        return f"<span class='low-score'>{score:.2f}</span>"

def main():
    # Set page title and header
    st.title("Yap.market Social Listening")
    st.markdown("This application demonstrates social listening capabilities for marketing campaigns.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("This tool analyzes social media content for relevance to marketing campaigns using AI. It measures how well tweets match campaign objectives and provides relevance scores (0-10).")
        st.markdown("The system uses OpenAI embeddings with optional reinforcement learning to improve scoring over time.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Campaign Analysis", "Batch Processing", "API Testing", "Scraper Integration"])
    
    # Tab 1: Campaign Analysis
    # [Your existing Campaign Analysis tab code]
    
    # Tab 2: Batch Processing
    # [Your existing Batch Processing tab code]
    
    # Tab 3: API Testing
    with tab3:
        st.header("API Integration Testing")
        
        with st.form("api_test_form"):
            test_content = st.text_area("Test Tweet Content", "Our new campaign is revolutionizing social listening!")
            test_api_key = st.text_input("API Key", value=os.getenv("INGESTION_API_KEY", ""))
            submitted = st.form_submit_button("Test API")
            
        if submitted:
            try:
                from datetime import datetime
                
                response = requests.post(
                    "http://localhost:8000/ingest-tweet",
                    headers={"X-API-Key": test_api_key},
                    json={
                        "tweet_id": "test_123",
                        "content": test_content,
                        "author_id": "test_user",
                        "campaign_id": "test_campaign",
                        "scraped_at": datetime.now().isoformat()
                    }
                )
                
                if response.status_code == 200:
                    st.success("API Test Successful!")
                    st.json(response.json())
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {str(e)}")
                
    # Tab 4: Scraper Integration
    with tab4:
        st.header("Twitter Scraper Integration")
        st.markdown("Scrape and analyze tweets directly from Twitter users")
        
        # Campaign brief input
        campaign_brief = st.text_area(
            "Campaign Brief:", 
            placeholder="Enter the marketing campaign brief here...", 
            height=150
        )
        
        # Twitter username input
        col1, col2 = st.columns([3, 1])
        with col1:
            twitter_username = st.text_input("Twitter Username (without @)", placeholder="elonmusk")
        with col2:
            max_tweets = st.number_input("Max Tweets", min_value=5, max_value=50, value=20)
        
        # Scrape and analyze button
        if st.button("Scrape & Analyze Tweets", type="primary"):
            if not campaign_brief:
                st.error("Please enter a campaign brief.")
            elif not twitter_username:
                st.error("Please enter a Twitter username.")
            else:
                with st.spinner(f"Scraping tweets from @{twitter_username}..."):
                    try:
                        # Create a data directory if it doesn't exist
                        os.makedirs("./data", exist_ok=True)
                        
                        # Create job ID for this scraping task
                        job_id = f"campaign-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        
                        # Prepare payload for scraper
                        scraper_payload = {
                            "job_id": job_id,
                            "username": twitter_username,
                            "max_tweets": max_tweets,
                            "campaign_brief": campaign_brief
                        }
                        
                        # Save job details
                        job_file = f"./data/{job_id}.json"
                        with open(job_file, "w") as f:
                            json.dump(scraper_payload, f)
                        
                        st.session_state["scraper_job_id"] = job_id
                        st.session_state["twitter_username"] = twitter_username
                        st.session_state["campaign_brief"] = campaign_brief
                        
                        # Try to call the scraper service
                        try:
                            response = requests.post(
                                "http://localhost:8000/api/scrape",
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {os.getenv('BACKEND_API_KEY', '')}"
                                },
                                json=scraper_payload,
                                timeout=5
                            )
                            
                            if response.status_code == 200:
                                st.success(f"Successfully requested tweets for @{twitter_username}")
                                st.info("Processing will continue in the background. Check results below.")
                            else:
                                st.warning(f"Scraper service returned: {response.text}")
                                st.error("Couldn't connect to the scraper service. It may not be running.")
                                st.code(
                                    "# Terminal 1: Start the Python API server\n" +
                                    "cd /Users/ayush/Documents/social-listening/yap-market\n" +
                                    "python api_server.py\n\n" +
                                    "# Terminal 2: Start the scraper service\n" +
                                    "cd /Users/ayush/Documents/social-listening/scraping-repo/scraper\n" +
                                    "bun run start",
                                    language="bash"
                                )
                                
                                # Manual analysis of sample tweets for demo purposes
                                st.info("Performing direct analysis of sample tweets...")
                                analyzer = SimpleSemanticAnalyzer()
                                
                                # Create sample tweets for demonstration
                                sample_tweets = [
                                    f"Just tried the new products from {twitter_username} and they're amazing! #recommended",
                                    f"Not impressed with the latest release from @{twitter_username}, expected more features.",
                                    f"The campaign from @{twitter_username} really speaks to me! Exactly what I was looking for!",
                                    f"Neutral thoughts about @{twitter_username}'s latest announcement. Let's see how it develops.",
                                    f"@{twitter_username} is changing the game with their innovative approach! #gamechangers"
                                ]
                                
                                # Analyze the sample tweets
                                results = []
                                for tweet in sample_tweets:
                                    score, context = analyzer.analyze_tweet(campaign_brief, tweet)
                                    results.append({
                                        "Tweet": tweet,
                                        "Score": score,
                                        "Context": context
                                    })
                                
                                # Create DataFrame
                                df = pd.DataFrame(results)
                                
                                # Display visualization
                                st.subheader("Sample Tweet Analysis Results")
                                col1, col2 = st.columns(2)
                                col1.metric("Average Score", f"{df['Score'].mean():.2f}/10")
                                col2.metric("Median Score", f"{df['Score'].median():.2f}/10")
                                
                                # Plot score distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(data=df, x="Score", bins=10, kde=True, ax=ax)
                                ax.set_title("Distribution of Tweet Relevance Scores")
                                ax.set_xlabel("Relevance Score (0-10)")
                                ax.set_ylabel("Count")
                                st.pyplot(fig)
                                
                                # Display tweets table sorted by score
                                st.subheader("Analyzed Tweets (Sorted by Relevance)")
                                st.dataframe(
                                    df.sort_values("Score", ascending=False),
                                    use_container_width=True
                                )
                        except requests.RequestException as e:
                            st.error(f"Could not connect to scraper service: {str(e)}")
                            st.info("Make sure the scraper service is running.")
                    except Exception as e:
                        st.error(f"Error during scraping process: {str(e)}")
        
        # Check for previous scraping jobs
        if "scraper_job_id" in st.session_state:
            job_id = st.session_state["scraper_job_id"]
            username = st.session_state.get("twitter_username", "")
            
            st.markdown("---")
            st.subheader("Previous Scraping Results")
            st.info(f"Showing results for job: {job_id}")
            
            # Check for analysis results files
            analysis_files = glob.glob(f"./data/{job_id}-*-analysis-*.json")
            
            if analysis_files:
                st.success("Found analysis results!")
                with open(analysis_files[0], "r") as f:
                    try:
                        analysis_results = json.load(f)
                        
                        # Create DataFrame from results
                        df = pd.DataFrame([
                            {"Tweet": item.get("tweet", ""), 
                             "Score": item.get("score", 0)} 
                            for item in analysis_results if "score" in item
                        ])
                        
                        if not df.empty:
                            # Display metrics
                            col1, col2 = st.columns(2)
                            col1.metric("Average Score", f"{df['Score'].mean():.2f}/10")
                            col2.metric("Median Score", f"{df['Score'].median():.2f}/10")
                            
                            # Plot score distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(data=df, x="Score", bins=10, kde=True, ax=ax)
                            ax.set_title("Distribution of Tweet Relevance Scores")
                            ax.set_xlabel("Relevance Score (0-10)")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                            
                            # Display tweets table
                            st.dataframe(
                                df.sort_values("Score", ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.warning("No scored tweets found in the results file.")
                    except json.JSONDecodeError:
                        st.error("Could not parse the results file. It may be corrupted.")
            else:
                st.info("No analysis results found yet. The scraper may still be processing your request.")
                st.info("Results will appear here once they're available.")
                
                # Create a refresh button
                if st.button("Refresh Results"):
                    st.experimental_rerun()

# FastAPI section
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# API setup
api_app = FastAPI(title="Yap.market Ingestion API")
api_key_header = APIKeyHeader(name="X-API-Key")

class ScrapedTweet(BaseModel):
    tweet_id: str
    content: str
    author_id: str
    campaign_id: str
    scraped_at: str

@api_app.post("/ingest-tweet")
async def ingest_tweet(
    tweet: ScrapedTweet,
    api_key: str = Security(api_key_header)
):
    """Endpoint for receiving scraped tweets from the TypeScript service"""
    if api_key != os.getenv("INGESTION_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Process tweet with existing semantic analyzer
    analyzer = SimpleSemanticAnalyzer()
    score, context = analyzer.analyze_tweet(
        campaign_brief="",  # Get actual brief from campaign ID
        tweet=tweet.content
    )
    
    # Store in database (implementation omitted)
    print(f"Processed tweet {tweet.tweet_id} with score {score}")
    return {"status": "processed", "score": score}

if __name__ == "__main__":
    main()
