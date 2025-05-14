import os
import sys
import json
import glob
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import openai
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv
# No need for sentence_transformers import - using only OpenAI embeddings

# Suppress specific NumPy warnings that occur during covariance calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

# Load environment variables from .env and then api_config.env
load_dotenv() # Loads .env by default

# Function to check OpenAI API key status
def check_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OpenAI API key not found in environment variables!")
        print("Please check your .env file and make sure it contains OPENAI_API_KEY=your_key_here")
        print(f"Current working directory: {os.getcwd()}")
        print(f".env file exists: {'Yes' if os.path.exists('.env') else 'No'}")
        
        # Try to read .env file directly to debug
        if os.path.exists('.env'):
            try:
                with open('.env', 'r') as f:
                    env_contents = f.read()
                    print(f".env file contents (first line masked):\n{'*' * 20}\n{env_contents.split('\n', 1)[1] if '\n' in env_contents else ''}")
            except Exception as e:
                print(f"Error reading .env file: {e}")
        return None
    else:
        # Mask the key for security
        masked_key = f"{api_key[:8]}...{api_key[-4:]}"
        print(f"✅ OpenAI API key loaded successfully: {masked_key}")
        return api_key

# Check and set OpenAI API key
api_key = check_openai_api_key()
openai.api_key = api_key
load_dotenv(dotenv_path="api_config.env", override=True) # Loads api_config.env and overrides if keys exist

# Simple version of semantic analyzer that doesn't depend on the full implementation
class LLMAnalyzer:
    """Analyzer that uses GPT-3.5-turbo to evaluate tweets against campaign briefs"""
    
    # Default system prompt as a class variable
    DEFAULT_SYSTEM_PROMPT = """
    You are an expert marketing analyst evaluating social media content against campaign briefs.
    
    You will be given a campaign brief and a tweet. Your task is to analyze the tweet and determine:
    
    1. RELEVANCE SCORE (0-10): How relevant the tweet is to the campaign brief
       - 0: Completely irrelevant, no connection to the campaign's themes, products, or audience
       - 5: Somewhat relevant, touches on related themes or audience interests
       - 10: Perfectly aligned with campaign objectives, messaging, and target audience
    
    2. SENTIMENT: The sentiment of the tweet in relation to the campaign
       - Positive: Favorable mentions, enthusiasm, support, or positive emotions
       - Neutral: Factual, informational, or balanced without clear emotional leaning
       - Negative: Critical, disappointed, frustrated, or negative emotions
    
    3. EXPLANATION: A brief, specific explanation of your scoring that references concrete elements
       from both the tweet and campaign brief
    
    4. INSIGHT: One actionable insight that could help improve the campaign based on this tweet
    
    Your analysis must be returned in this exact JSON format:
    {"relevance_score": float, "sentiment": string, "explanation": string, "insight": string}
    
    Be objective and precise. Focus on how well the tweet aligns with the campaign's specific goals,
    target audience characteristics, and key messaging points as described in the brief.
    """
    
    def __init__(self, api_key=None, custom_system_prompt=None, model_name=None, embedding_model_name=None):
        """Initialize the LLM analyzer with OpenAI API key and optional custom system prompt"""
        # Try to get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or openai.api_key
        
        # Set system prompt (use custom if provided, otherwise use default)
        self.system_prompt = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Set model names (use custom if provided, otherwise use defaults)
        self.model = model_name or "gpt-3.5-turbo"
        self.embedding_model = embedding_model_name or "text-embedding-ada-002"
        
        # Debug API key information
        if self.api_key:
            masked_key = f"{self.api_key[:8]}...{self.api_key[-5:]}"
            print(f"✅ LLMAnalyzer: API key found (starts with {self.api_key[:8]}...)")
            print(f"Using LLM model: {self.model}")
            print(f"Using embedding model: {self.embedding_model}")
        else:
            print("❌ LLMAnalyzer: No API key found! Check your .env file or pass a key directly.")
            print(f"Current environment variables: {list(os.environ.keys())}")
            print(f"Is OPENAI_API_KEY in environment? {'Yes' if 'OPENAI_API_KEY' in os.environ else 'No'}")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def analyze_tweet_with_llm(self, campaign_brief, tweet_text):
        """Analyze a tweet using GPT-3.5-turbo to evaluate relevance and sentiment"""
        print("\n" + "*"*50)
        print("STARTING LLM ANALYSIS OF TWEET")
        print("*"*50)
        
        if not tweet_text or not campaign_brief:
            print("Empty input provided to LLM analyzer")
            return {
                "relevance_score": 0.0,
                "sentiment": "neutral",
                "explanation": "Empty input",
                "insight": ""
            }
        
        # Clean inputs
        campaign_brief = campaign_brief.strip()
        tweet_text = tweet_text.strip()
        
        print(f"Tweet to analyze: {tweet_text[:100]}...")
        print(f"Campaign brief length: {len(campaign_brief)} characters")
        
        # Use the system prompt from the instance (which could be custom or default)
        system_prompt = self.system_prompt
        print(f"Using {'custom' if self.system_prompt != self.DEFAULT_SYSTEM_PROMPT else 'default'} system prompt")
        
        # User message combining campaign brief and tweet
        user_message = f"""CAMPAIGN BRIEF:\n{campaign_brief}\n\nTWEET TO ANALYZE:\n{tweet_text}"""
        
        try:
            print("Calling OpenAI API...")
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            print(f"LLM response received: {result_text[:200]}...")
            
            try:
                result = json.loads(result_text)
                
                # Ensure the score is within 0-10 range
                if "relevance_score" in result:
                    result["relevance_score"] = max(0, min(10, float(result["relevance_score"])))
                else:
                    result["relevance_score"] = 0.0
                    
                # Ensure sentiment is one of the expected values
                if "sentiment" in result:
                    sentiment = result["sentiment"].lower()
                    if sentiment not in ["positive", "negative", "neutral"]:
                        result["sentiment"] = "neutral"
                else:
                    result["sentiment"] = "neutral"
                
                print(f"Analysis complete - Score: {result['relevance_score']}, Sentiment: {result['sentiment']}")
                print("*"*50 + "\n")
                return result
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from LLM response: {str(e)}")
                print(f"Raw response: {result_text}")
                return {
                    "relevance_score": 0.0,
                    "sentiment": "neutral",
                    "explanation": f"Error parsing LLM response",
                    "insight": "Consider reviewing the campaign brief format"
                }
            
        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "relevance_score": 0.0,
                "sentiment": "neutral",
                "explanation": f"Error: {str(e)}",
                "insight": "Check API key and network connection"
            }


class SimpleSemanticAnalyzer:
    """Simple analyzer that uses OpenAI embeddings to evaluate tweets against campaign briefs"""
    
    def __init__(self, api_key=None, embedding_model_name=None):
        """Initialize the semantic analyzer with OpenAI API key and optional embedding model name"""
        # Try to get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or openai.api_key
        
        # Set embedding model name (use custom if provided, otherwise use default)
        self.model = embedding_model_name or "text-embedding-ada-002"
        
        # Debug API key information
        if self.api_key:
            print(f"✅ SimpleSemanticAnalyzer: API key found")
            print(f"Using embedding model: {self.model}")
        else:
            print("❌ SimpleSemanticAnalyzer: No API key found! Check your .env file or pass a key directly.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def get_embeddings(self, text):
        """Get embeddings for text using OpenAI API"""
        if not text or not text.strip():
            print("Empty text provided to get_embeddings")
            return None
        
        try:
            print(f"Calling OpenAI API with key ending in ...{self.api_key[-5:] if self.api_key else 'None'}")
            print(f"Using model: {self.model}")
            print(f"Input text (first 50 chars): {text[:50]}...")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Check if we got a valid response
            if not response or not hasattr(response, 'data') or not response.data:
                print("Invalid response from OpenAI API: No data returned")
                return None
                
            embedding = response.data[0].embedding
            print(f"Successfully got embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"ERROR getting embedding: {str(e)}")
            import traceback
            traceback.print_exc()
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
        print("\n" + "="*50)
        print("STARTING TWEET ANALYSIS")
        print("="*50)
        
        # Check for empty inputs
        if not tweet or not campaign_brief:
            print(f"⚠️ Missing input: campaign_brief empty: {not campaign_brief}, tweet empty: {not tweet}")
            return 0.0, "Empty input"
            
        # Clean input
        campaign_brief = campaign_brief.strip() if isinstance(campaign_brief, str) else ""
        tweet = tweet.strip() if isinstance(tweet, str) else ""
        
        print(f"Campaign brief length: {len(campaign_brief)} characters")
        print(f"Tweet length: {len(tweet)} characters")
        
        if not tweet or not campaign_brief:
            print(f"⚠️ Empty after cleaning: campaign_brief empty: {not campaign_brief}, tweet empty: {not tweet}")
            return 0.0, "Empty input after cleaning"
        
        try:
            print(f"\nAnalyzing tweet: {tweet[:100]}...")
            print(f"Using campaign brief: {campaign_brief[:100]}...")
            
            # Verify API key
            if not self.api_key:
                print("⚠️ NO API KEY FOUND! Make sure OPENAI_API_KEY is set in your environment.")
                return 0.0, "No OpenAI API key found"
            
            # Get embeddings
            print("\nGetting campaign brief embedding...")
            campaign_embedding = self.get_embeddings(campaign_brief)
            print("\nGetting tweet embedding...")
            tweet_embedding = self.get_embeddings(tweet)
            
            # Check if embeddings were successfully retrieved
            if not campaign_embedding:
                print("⚠️ Failed to get campaign brief embedding")
            if not tweet_embedding:
                print("⚠️ Failed to get tweet embedding")
                
            if not campaign_embedding or not tweet_embedding:
                print("⚠️ Failed to get embeddings - returning score 0")
                return 0.0, "Failed to get embeddings"
            
            # Calculate similarity
            print("\nCalculating similarity...")
            similarity = self.calculate_similarity(campaign_embedding, tweet_embedding)
            print(f"Similarity calculated: {similarity}")
            
            # Scale to 0-10 range
            score = similarity * 10.0
            context = f"Relevance based on: CoralApp features and messaging match: {similarity:.3f}"
            
            print(f"Final score: {score:.2f}/10")
            print("="*50)
            print("ANALYSIS COMPLETE")
            print("="*50 + "\n")
            return score, context
        except Exception as e:
            error_msg = f"⚠️ Error analyzing tweet: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
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
    
    # Create a single tab for Twitter Analysis
    tab1 = st.tabs(["Twitter Analysis"])[0]
    
    # Twitter Analysis tab (formerly Scraper Integration)
    with tab1:
        st.header("Twitter Analysis")
        st.markdown("Analyze tweets from Twitter users - either by scraping or manual input")
        
        # Campaign brief input
        campaign_brief = st.text_area(
            "Campaign Brief:", 
            placeholder="Enter the marketing campaign brief here...", 
            height=150
        )
        
        # Option to choose between scraping or manual input
        analysis_method = st.radio(
            "Choose analysis method:",
            ["Scrape tweets automatically", "Enter tweets manually"],
            horizontal=True
        )
        
        if analysis_method == "Scrape tweets automatically":
            # Twitter username input
            twitter_username = st.text_input("Twitter Username (without @)", value="")
            
            # Max tweets slider/input
            max_tweets = st.number_input("Max Tweets", min_value=5, max_value=200, value=50, step=5)
            st.caption("Number of tweets to scrape (higher values may take longer)")
            
            # Scrape and analyze button
            if st.button("Scrape & Analyze Tweets", type="primary"):
                if not campaign_brief:
                    st.error("Please enter a campaign brief.")
                elif not twitter_username:
                    st.error("Please enter a Twitter username.")
                else:
                    with st.spinner(f"Scraping tweets from @{twitter_username}..."):
                        try:
                            # Import datetime here to ensure it's accessible in this scope
                            from datetime import datetime
                            
                            # Create a data directory if it doesn't exist
                            os.makedirs("./data", exist_ok=True)
                            
                            # Generate a consistent campaign ID based on username and date
                            # This ensures we can find results from the same username on the same day
                            today = datetime.now().strftime('%Y%m%d')
                            campaign_id = f"{twitter_username.lower()}-{today}"
                            
                            # Save the campaign brief
                            with open(f"./data/{campaign_id}.json", "w") as f:
                                json.dump({"campaign_brief": campaign_brief}, f)
                            
                            # Store campaign brief in session state for later use
                            st.session_state["campaign_brief"] = campaign_brief
                            
                            # Call the Twitter scraper API
                            scraper_url = "http://localhost:3000/scrape"
                            payload = {
                                "username": twitter_username,
                                "maxTweets": max_tweets,
                                "campaignId": campaign_id
                            }
                            
                            try:
                                response = requests.post(scraper_url, json=payload)
                                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                                
                                # Check if the scraper returned any tweets
                                if response.status_code == 200:
                                    result = response.json()
                                    tweet_count = result.get("tweetCount", 0)
                                    
                                    if tweet_count > 0:
                                        st.success(f"Successfully scraped {tweet_count} tweets from @{twitter_username}!")
                                        
                                        # Store the job ID in session state
                                        st.session_state["job_id"] = campaign_id
                                        
                                        # Add a button to view results
                                        if st.button("View Analysis Results"):
                                            st.session_state["active_tab"] = "results"
                                            st.experimental_rerun()
                                    else:
                                        st.warning(f"No tweets found for @{twitter_username}. Try a different username or increase the max tweets.")
                                else:
                                    st.error(f"Error from scraper API: {response.text}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Could not connect to the scraper API: {str(e)}")
                                st.info("Make sure the Twitter scraper is running on port 3000.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        else:  # Manual tweet input
            # Number of tweets to analyze
            num_tweets = st.number_input("Number of tweets to analyze", min_value=1, max_value=20, value=3)
            
            # Create input fields for each tweet
            manual_tweets = []
            for i in range(num_tweets):
                tweet_text = st.text_area(
                    f"Tweet {i+1}", 
                    placeholder="Enter tweet content here...",
                    height=100,
                    key=f"tweet_{i}"
                )
                manual_tweets.append(tweet_text)
            
            # Advanced options section with system prompt editor, API key, and model selection
            with st.expander("Advanced Options"):
                # API Key section
                st.subheader("OpenAI API Configuration")
                
                # Initialize API key in session state if not already there
                if "openai_api_key" not in st.session_state:
                    st.session_state["openai_api_key"] = ""
                
                # API key input - let the user enter it without pre-filling
                custom_api_key = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state["openai_api_key"],
                    type="password",
                    placeholder="Enter your OpenAI API key here...",
                    help="Enter your OpenAI API key. If left empty, the app will try to use the key from the .env file."
                )
                
                # Save the API key to session state
                st.session_state["openai_api_key"] = custom_api_key
                
                # Model selection
                st.subheader("Model Selection")
                
                # Initialize model names in session state if not already there
                if "llm_model" not in st.session_state:
                    st.session_state["llm_model"] = "gpt-3.5-turbo"
                if "embedding_model" not in st.session_state:
                    st.session_state["embedding_model"] = "text-embedding-ada-002"
                
                # Model selection dropdowns
                col1, col2 = st.columns(2)
                
                with col1:
                    llm_model = st.selectbox(
                        "LLM Model",
                        options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                        index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"].index(st.session_state["llm_model"]),
                        help="Select the OpenAI model to use for detailed tweet analysis"
                    )
                    st.session_state["llm_model"] = llm_model
                
                with col2:
                    embedding_model = st.selectbox(
                        "Embedding Model",
                        options=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                        index=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"].index(st.session_state["embedding_model"]),
                        help="Select the OpenAI model to use for semantic embeddings"
                    )
                    st.session_state["embedding_model"] = embedding_model
                
                # System prompt section
                st.subheader("LLM System Prompt")
                st.markdown("Customize the instructions given to the AI model for tweet analysis.")
                
                # Initialize the system prompt in session state if not already there
                if "system_prompt" not in st.session_state:
                    st.session_state["system_prompt"] = LLMAnalyzer.DEFAULT_SYSTEM_PROMPT
                
                # System prompt editor
                custom_system_prompt = st.text_area(
                    "Edit System Prompt",
                    value=st.session_state["system_prompt"],
                    height=400,
                    key="system_prompt_editor"
                )
                
                # Save the custom prompt to session state
                st.session_state["system_prompt"] = custom_system_prompt
                
                # Reset buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reset to Default Prompt"):
                        st.session_state["system_prompt"] = LLMAnalyzer.DEFAULT_SYSTEM_PROMPT
                        st.experimental_rerun()
                with col2:
                    if st.button("Reset All Settings"):
                        st.session_state["system_prompt"] = LLMAnalyzer.DEFAULT_SYSTEM_PROMPT
                        st.session_state["llm_model"] = "gpt-3.5-turbo"
                        st.session_state["embedding_model"] = "text-embedding-ada-002"
                        st.session_state["openai_api_key"] = ""
                        st.experimental_rerun()
            
            # Analyze button for manual tweets
            if st.button("Analyze Manual Tweets", type="primary"):
                if not campaign_brief:
                    st.error("Please enter a campaign brief.")
                elif not any(tweet.strip() for tweet in manual_tweets):
                    st.error("Please enter at least one tweet.")
                else:
                    with st.spinner("Analyzing manual tweets..."):
                        try:
                            # Import datetime here to ensure it's accessible in this scope
                            from datetime import datetime
                            
                            # Create a data directory if it doesn't exist
                            os.makedirs("./data", exist_ok=True)
                            
                            # Generate a campaign ID for manual tweets and set a simulated username
                            today = datetime.now().strftime('%Y%m%d')
                            # Simulate a Twitter username for display purposes only (not sent to scraper)
                            twitter_username = "manual_analysis"
                            campaign_id = f"manual-{today}"
                            
                            # Save the campaign brief
                            with open(f"./data/{campaign_id}.json", "w") as f:
                                json.dump({"campaign_brief": campaign_brief}, f)
                            
                            # Store campaign brief in session state for later use
                            st.session_state["campaign_brief"] = campaign_brief
                            
                            # Process each tweet with the semantic analyzer
                            # Use custom settings if available
                            custom_prompt = st.session_state.get("system_prompt", LLMAnalyzer.DEFAULT_SYSTEM_PROMPT)
                            custom_api_key = st.session_state.get("openai_api_key", None)
                            llm_model = st.session_state.get("llm_model", "gpt-3.5-turbo")
                            embedding_model = st.session_state.get("embedding_model", "text-embedding-ada-002")
                            
                            # Initialize analyzers with custom settings
                            analyzer = SimpleSemanticAnalyzer(api_key=custom_api_key, embedding_model_name=embedding_model)
                            llm_analyzer = LLMAnalyzer(
                                api_key=custom_api_key,
                                custom_system_prompt=custom_prompt,
                                model_name=llm_model,
                                embedding_model_name=embedding_model
                            )
                            results = []
                            
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            
                            # Process each tweet
                            for i, tweet_text in enumerate(manual_tweets):
                                if tweet_text.strip():  # Skip empty tweets
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(manual_tweets))
                                    
                                    # First get basic semantic score
                                    score, context = analyzer.analyze_tweet(campaign_brief, tweet_text)
                                    
                                    # Then use LLM analyzer with custom system prompt for detailed analysis
                                    llm_result = None
                                    try:
                                        llm_result = llm_analyzer.analyze_tweet_with_llm(campaign_brief, tweet_text)
                                        st.write(f"LLM analysis for tweet {i+1} completed successfully")
                                    except Exception as e:
                                        st.error(f"LLM analysis failed: {str(e)}")
                                    
                                    # Create a result object similar to scraped tweets
                                    result = {
                                        "tweet_id": f"manual-{i+1}",
                                        "content": tweet_text,
                                        "author_id": "manual",
                                        "campaign_id": campaign_id,
                                        "scraped_at": datetime.now().isoformat(),
                                        "analyzed_at": datetime.now().isoformat(),
                                        "score": float(score),
                                        "context": context,
                                        "llm_analysis": llm_result if llm_result else None
                                    }
                                    
                                    results.append(result)
                            
                            # Clear progress bar
                            progress_bar.empty()
                            
                            # Save results to file
                            results_file = f"./data/{campaign_id}-realtime-results.json"
                            with open(results_file, "w") as f:
                                json.dump(results, f)
                            
                            # Store the job ID in session state (use scraper_job_id to be consistent with the results tab)
                            st.session_state["scraper_job_id"] = campaign_id
                            st.session_state["job_id"] = campaign_id  # Keep this for backward compatibility
                            
                            # Success message
                            st.success(f"Successfully analyzed {len(results)} tweets!")
                            
                            # Add a button to view results
                            if st.button("View Analysis Results", key="view_manual_results"):
                                # Switch to the results tab
                                st.session_state["active_tab"] = "results"
                                # Make sure we're using the right session state variable for the job ID
                                st.session_state["current_job_id"] = campaign_id
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error analyzing tweets: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                        
                        # Manual tweet analysis is complete
                        
        # Check for previous scraping jobs
        if "scraper_job_id" in st.session_state:
            job_id = st.session_state["scraper_job_id"]
            username = st.session_state.get("twitter_username", "")
            
            st.markdown("---")
            st.subheader("Previous Scraping Results")
            st.info(f"Showing results for job: {job_id}")
            
            # Debug information
            st.code(f"Session state: {st.session_state}")
            st.code(f"Looking for realtime file: ./data/{job_id}-realtime-results.json")
            
            # Create a container for real-time updates
            results_container = st.container()
            
            # Add auto-refresh capability
            auto_refresh = st.checkbox("Auto-refresh results", value=True)
            
            # Create a placeholder for the results
            with results_container:
                # First check for real-time results file
                realtime_file = f"./data/{job_id}-realtime-results.json"
                if os.path.exists(realtime_file):
                    try:
                        with open(realtime_file, "r") as f:
                            realtime_results = json.load(f)
                        
                        if realtime_results:
                            st.success(f"Found {len(realtime_results)} analyzed tweets so far")
                            
                            # Create DataFrame from results
                            df = pd.DataFrame([
                                {
                                    "Tweet": item.get("content", ""), 
                                    "Score": item.get("score", 0),
                                    "Analyzed At": item.get("analyzed_at", ""),
                                    "Context": item.get("context", "")
                                } 
                                for item in realtime_results
                            ])
                            
                            if not df.empty:
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Average Score", f"{df['Score'].mean():.2f}/10")
                                col2.metric("Median Score", f"{df['Score'].median():.2f}/10")
                                col3.metric("Tweets Analyzed", f"{len(df)}")
                                
                                # Plot score distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(data=df, x="Score", bins=10, kde=True, ax=ax)
                                ax.set_title("Distribution of Tweet Relevance Scores")
                                ax.set_xlabel("Relevance Score (0-10)")
                                ax.set_ylabel("Count")
                                st.pyplot(fig)
                                
                                # Add LLM processing section with threshold slider
                                st.subheader("LLM Processing")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    threshold = st.slider(
                                        "Score threshold for LLM processing", 
                                        min_value=0.0, 
                                        max_value=10.0, 
                                        value=5.0, 
                                        step=0.5,
                                        help="Only tweets with scores above this threshold will be processed by the LLM"
                                    )
                                with col2:
                                    pass_to_llm = st.button("Pass to LLM", type="primary")
                                
                                # Count eligible tweets
                                eligible_tweets = df[df['Score'] >= threshold]
                                st.info(f"{len(eligible_tweets)} tweets meet the threshold criteria of {threshold}+")
                                
                                # Handle LLM processing
                                if pass_to_llm:
                                    if len(eligible_tweets) > 0:
                                        # Create a container for the LLM processing status
                                        llm_status_container = st.empty()
                                        llm_status_container.info(f"Preparing to process {len(eligible_tweets)} tweets with LLM...")
                                        
                                        # Get campaign brief from multiple possible sources
                                        campaign_brief = ""
                                        
                                        # Try multiple possible file locations for the campaign brief
                                        possible_files = [
                                            f"./data/{job_id}.json",  # Standard location
                                            f"./data/{job_id.replace('-', '_')}.json",  # With underscores instead of hyphens
                                            "./data/test_campaign.json",  # Default test campaign
                                            "./campaign_brief.txt"  # Raw text file
                                        ]
                                        
                                        brief_found = False
                                        for file_path in possible_files:
                                            if os.path.exists(file_path):
                                                try:
                                                    if file_path.endswith('.json'):
                                                        with open(file_path, "r") as f:
                                                            brief_data = json.load(f)
                                                            if isinstance(brief_data, dict) and "campaign_brief" in brief_data:
                                                                campaign_brief = brief_data.get("campaign_brief", "")
                                                                llm_status_container.info(f"Loaded campaign brief from {file_path}")
                                                                brief_found = True
                                                                break
                                                    else:
                                                        with open(file_path, "r") as f:
                                                            campaign_brief = f.read()
                                                            llm_status_container.info(f"Loaded campaign brief from {file_path}")
                                                            brief_found = True
                                                            break
                                                except Exception as e:
                                                    print(f"Error loading campaign brief from {file_path}: {str(e)}")
                                        
                                        # If no brief found, try session state or use default
                                        if not brief_found:
                                            campaign_brief = st.session_state.get("campaign_brief", "")
                                            if campaign_brief:
                                                llm_status_container.info("Using campaign brief from session state")
                                            else:
                                                # Use a default brief as last resort
                                                campaign_brief = """CoralApp Campaign Brief

CoralApp is a new social media platform designed to connect users with shared interests in a more meaningful way. Our app focuses on creating deeper connections through content curation and interest-based communities.

Key features include interest-based communities, AI-powered content curation, privacy controls, and an intuitive interface.

Target audience: Tech-savvy young adults (18-35) looking for meaningful connections."""
                                                llm_status_container.warning("Using default CoralApp brief as no campaign brief was found")
                                        
                                        with st.spinner(f"Processing {len(eligible_tweets)} tweets with LLM..."):
                                            # Process tweets with GPT-3.5-turbo
                                            llm_analyzer = LLMAnalyzer()
                                            llm_results = []
                                            insights = []
                                            
                                            # Process each eligible tweet
                                            progress_bar = st.progress(0)
                                            for i, (_, row) in enumerate(eligible_tweets.iterrows()):
                                                # Update progress and status message
                                                progress_bar.progress((i + 1) / len(eligible_tweets))
                                                llm_status_container.info(f"Processing tweet {i+1} of {len(eligible_tweets)}...")
                                                
                                                # Analyze tweet with LLM
                                                tweet_text = row['Tweet']
                                                result = llm_analyzer.analyze_tweet_with_llm(campaign_brief, tweet_text)
                                                
                                                # Add to results
                                                llm_results.append({
                                                    "tweet": tweet_text,
                                                    "original_score": row['Score'],
                                                    "llm_score": result["relevance_score"],
                                                    "sentiment": result["sentiment"],
                                                    "explanation": result["explanation"],
                                                    "insight": result.get("insight", "")
                                                })
                                                
                                                # Collect insights
                                                if "insight" in result and result["insight"]:
                                                    insights.append(result["insight"])
                                            
                                            # Clear status container
                                            llm_status_container.empty()
                                            
                                            # Deduplicate and limit insights
                                            unique_insights = list(set(insights))
                                            top_insights = unique_insights[:5] if len(unique_insights) > 5 else unique_insights
                                            
                                            # Store results in session state without datetime
                                            st.session_state["llm_processed"] = True
                                            st.session_state["llm_results"] = {
                                                "threshold": threshold,
                                                "tweets_processed": len(eligible_tweets),
                                                "detailed_results": llm_results,
                                                "sample_insights": top_insights if top_insights else [
                                                    "No specific insights generated from the analyzed tweets."
                                                ]
                                            }
                                            
                                            # Clear progress bar
                                            progress_bar.empty()
                                            
                                        st.success("LLM processing complete!")
                                    else:
                                        st.warning("No tweets meet the threshold criteria. Adjust the threshold or analyze more tweets.")
                                
                                # Display LLM results if available
                                if st.session_state.get("llm_processed"):
                                    st.header("🤖 LLM Analysis Results")
                                    results = st.session_state.get("llm_results", {})
                                    
                                    # Display processing metadata in a cleaner format
                                    meta_cols = st.columns(2)
                                    meta_cols[0].metric("Tweets Analyzed", f"{results.get('tweets_processed')}")
                                    meta_cols[1].metric("Threshold Used", f"{results.get('threshold')}/10")
                                    
                                    # Display insights from LLM analysis
                                    st.subheader("📊 Key Campaign Insights")
                                    insights = results.get("sample_insights", [])
                                    if insights:
                                        for i, insight in enumerate(insights):
                                            st.markdown(f"**{i+1}.** {insight}")
                                    else:
                                        st.info("No specific insights were generated from the analyzed tweets.")
                                    
                                    # Display sentiment distribution
                                    detailed_results = results.get("detailed_results", [])
                                    if detailed_results:
                                        st.subheader("😊 Sentiment Analysis")
                                        
                                        # Create sentiment counts
                                        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
                                        for result in detailed_results:
                                            sentiment = result.get("sentiment", "neutral").lower()
                                            if sentiment in sentiment_counts:
                                                sentiment_counts[sentiment] += 1
                                        
                                        # Calculate percentages
                                        total = sum(sentiment_counts.values())
                                        sentiment_percentages = {
                                            k: (v/total*100) for k, v in sentiment_counts.items()
                                        }
                                        
                                        # Display sentiment metrics with color coding
                                        cols = st.columns(3)
                                        cols[0].metric("Positive 😊", sentiment_counts["positive"], 
                                                 f"{sentiment_percentages['positive']:.1f}%")
                                        cols[1].metric("Neutral 😐", sentiment_counts["neutral"],
                                                 f"{sentiment_percentages['neutral']:.1f}%")
                                        cols[2].metric("Negative 😞", sentiment_counts["negative"],
                                                 f"{sentiment_percentages['negative']:.1f}%")
                                        
                                        # Create a pie chart for sentiment distribution
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        colors = ['#4CAF50', '#9E9E9E', '#F44336']
                                        wedges, texts, autotexts = ax.pie(
                                            sentiment_counts.values(), 
                                            labels=None,
                                            autopct='%1.1f%%',
                                            startangle=90,
                                            colors=colors
                                        )
                                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                                        plt.legend(wedges, [f"{k.capitalize()} ({v})" for k, v in sentiment_counts.items()],
                                                  title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                                        st.pyplot(fig)
                                        
                                        # Create a DataFrame for the detailed results
                                        llm_df = pd.DataFrame(detailed_results)
                                        
                                        # Display score comparison chart if we have both scores
                                        if "original_score" in llm_df.columns and "llm_score" in llm_df.columns:
                                            st.subheader("📈 Score Comparison")
                                            
                                            # Create columns for metrics
                                            score_cols = st.columns(2)
                                            
                                            # Calculate average scores and deltas
                                            avg_original = llm_df["original_score"].mean()
                                            avg_llm = llm_df["llm_score"].mean()
                                            delta = avg_llm - avg_original
                                            
                                            # Display metrics with comparison
                                            score_cols[0].metric(
                                                "Avg. Embedding Score", 
                                                f"{avg_original:.2f}/10", 
                                                delta=None
                                            )
                                            score_cols[1].metric(
                                                "Avg. LLM Score", 
                                                f"{avg_llm:.2f}/10", 
                                                delta=f"{delta:+.2f}"
                                            )
                                            
                                            # Create bar chart
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            x = ["Embedding Score", "LLM Score"]
                                            y = [avg_original, avg_llm]
                                            bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e'])
                                            ax.set_ylim(0, 10)
                                            ax.set_ylabel("Average Score (0-10)")
                                            ax.set_title("Comparison of Embedding vs. LLM Scoring Methods")
                                            
                                            # Add value labels on top of bars
                                            for i, v in enumerate(y):
                                                ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
                                                
                                            st.pyplot(fig)
                                            
                                            # Create a scatter plot comparing scores
                                            st.subheader("📏 Score Distribution Comparison")
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.scatter(llm_df["original_score"], llm_df["llm_score"], alpha=0.6)
                                            
                                            # Add diagonal line for reference (perfect correlation)
                                            ax.plot([0, 10], [0, 10], 'k--', alpha=0.3)
                                            
                                            # Add labels and title
                                            ax.set_xlabel("Embedding Score")
                                            ax.set_ylabel("LLM Score")
                                            ax.set_xlim(0, 10)
                                            ax.set_ylim(0, 10)
                                            ax.set_title("Embedding vs. LLM Score Correlation")
                                            ax.grid(True, alpha=0.3)
                                            
                                            # Calculate and display correlation
                                            corr = llm_df["original_score"].corr(llm_df["llm_score"])
                                            ax.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
                                            
                                            st.pyplot(fig)
                                            
                                    # Display detailed tweet analysis table
                                    st.subheader("💬 Tweet Analysis Details")
                                    
                                    if not llm_df.empty:
                                        # Prepare the dataframe for display
                                        display_df = llm_df.copy()
                                        
                                        # Rename columns for better display
                                        display_df = display_df.rename(columns={
                                            "tweet": "Tweet",
                                            "original_score": "Embedding Score",
                                            "llm_score": "LLM Score",
                                            "sentiment": "Sentiment",
                                            "explanation": "Explanation"
                                        })
                                        
                                        # Format scores to 2 decimal places
                                        if "Embedding Score" in display_df.columns:
                                            display_df["Embedding Score"] = display_df["Embedding Score"].apply(lambda x: f"{x:.2f}")
                                        if "LLM Score" in display_df.columns:
                                            display_df["LLM Score"] = display_df["LLM Score"].apply(lambda x: f"{x:.2f}")
                                            
                                        # Add sentiment emoji column
                                        if "Sentiment" in display_df.columns:
                                            sentiment_emojis = {
                                                "positive": "😊",
                                                "neutral": "😐",
                                                "negative": "😞"
                                            }
                                            display_df["Sentiment"] = display_df["Sentiment"].apply(
                                                lambda x: f"{x.capitalize()} {sentiment_emojis.get(x.lower(), '')}"
                                            )
                                        
                                        # Select columns to display
                                        cols_to_display = ["Tweet", "Embedding Score", "LLM Score", "Sentiment", "Explanation"]
                                        display_df = display_df[[col for col in cols_to_display if col in display_df.columns]]
                                        
                                        # Add an expander for the detailed table
                                        with st.expander("View detailed tweet analysis", expanded=False):
                                            # Create a custom CSS to limit tweet length in the table
                                            st.markdown("""
                                            <style>
                                            .tweet-cell { 
                                                max-width: 300px; 
                                                overflow: hidden; 
                                                text-overflow: ellipsis; 
                                                white-space: nowrap; 
                                            }
                                            </style>
                                            """, unsafe_allow_html=True)
                                            
                                            # Display the table
                                            st.dataframe(display_df, use_container_width=True)
                                            
                                            # Add download button for CSV export
                                            csv = display_df.to_csv(index=False)
                                            st.download_button(
                                                label="Download analysis as CSV",
                                                data=csv,
                                                file_name="tweet_analysis.csv",
                                                mime="text/csv"
                                            )
                                    else:
                                        st.info("No tweet analysis data available.")
                                        
                                    # Add interactive tweet viewer with full details
                                    if not llm_df.empty:
                                        st.subheader("🔍 Tweet Inspector")
                                        
                                        # Create a tweet selector
                                        selected_tweet_index = st.selectbox(
                                            "Select a tweet to inspect",
                                            options=list(range(len(llm_df))),
                                            format_func=lambda i: f"Tweet {i+1}: {llm_df.iloc[i]['tweet'][:50]}..."
                                        )
                                        
                                        # Display the selected tweet details
                                        if selected_tweet_index is not None:
                                            tweet_data = llm_df.iloc[selected_tweet_index]
                                            
                                            # Create columns for tweet details
                                            col1, col2 = st.columns([2, 1])
                                            
                                            with col1:
                                                st.markdown("### Tweet Content")
                                                st.markdown(f"""<div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6;">
                                                {tweet_data['tweet']}
                                                </div>""", unsafe_allow_html=True)
                                                
                                                st.markdown("### LLM Analysis")
                                                st.markdown(f"**Explanation:** {tweet_data['explanation']}")
                                                if 'insight' in tweet_data and tweet_data['insight']:
                                                    st.markdown(f"**Insight:** {tweet_data['insight']}")
                                            
                                            with col2:
                                                # Show scores and sentiment
                                                st.markdown("### Scores")
                                                
                                                # Embedding score
                                                if 'original_score' in tweet_data:
                                                    st.metric("Embedding Score", f"{tweet_data['original_score']:.2f}/10")
                                                
                                                # LLM score
                                                if 'llm_score' in tweet_data:
                                                    st.metric("LLM Score", f"{tweet_data['llm_score']:.2f}/10")
                                                
                                                # Sentiment with emoji
                                                if 'sentiment' in tweet_data:
                                                    sentiment = tweet_data['sentiment'].lower()
                                                    emoji = {
                                                        "positive": "😊",
                                                        "neutral": "😐",
                                                        "negative": "😞"
                                                    }.get(sentiment, "")
                                                    
                                                    st.markdown(f"### Sentiment: {sentiment.capitalize()} {emoji}")
                                                    
                                                    # Sentiment color bar
                                                    sentiment_color = {
                                                        "positive": "#4CAF50",
                                                        "neutral": "#9E9E9E",
                                                        "negative": "#F44336"
                                                    }.get(sentiment, "#9E9E9E")
                                                    
                                                    st.markdown(f"""
                                                    <div style="background-color: {sentiment_color}; height: 20px; border-radius: 5px;"></div>
                                                    """, unsafe_allow_html=True)
                                        
                                    # Display detailed LLM analysis
                                    if detailed_results:
                                        st.subheader("Detailed LLM Analysis")
                                        for i, result in enumerate(detailed_results):
                                            with st.expander(f"Tweet {i+1}: LLM Score {result.get('llm_score', 0):.2f}/10 - {result.get('sentiment', 'neutral').title()}"):
                                                st.write(f"**Tweet:** {result.get('tweet', '')}")
                                                st.write(f"**Original Score:** {result.get('original_score', 0):.2f}/10")
                                                st.write(f"**LLM Score:** {result.get('llm_score', 0):.2f}/10")
                                                
                                                # Display sentiment with color
                                                sentiment = result.get('sentiment', 'neutral').lower()
                                                if sentiment == 'positive':
                                                    st.success(f"**Sentiment:** {sentiment.title()}")
                                                elif sentiment == 'negative':
                                                    st.error(f"**Sentiment:** {sentiment.title()}")
                                                else:
                                                    st.info(f"**Sentiment:** {sentiment.title()}")
                                                    
                                                st.write(f"**Explanation:** {result.get('explanation', '')}")
                                                if result.get('insight'):
                                                    st.write(f"**Insight:** {result.get('insight', '')}")
                                    
                                
                                # Display tweets table with expanded view
                                st.subheader("Analyzed Tweets")
                                for i, row in df.sort_values("Score", ascending=False).iterrows():
                                    # Highlight tweets that meet the threshold
                                    meets_threshold = row['Score'] >= threshold
                                    expander_label = f"Tweet {i+1}: Score {row['Score']:.2f}/10"
                                    if meets_threshold:
                                        expander_label += " ✅"
                                    
                                    with st.expander(expander_label):
                                        st.write(f"**Tweet Text:** {row['Tweet']}")
                                        st.write(f"**Relevance Context:** {row['Context']}")
                                        st.write(f"**Analyzed At:** {row['Analyzed At']}")
                                        if meets_threshold:
                                            st.success("This tweet meets the threshold criteria for LLM processing")
                                        else:
                                            st.info("This tweet does not meet the threshold criteria")
                                
                            else:
                                st.warning("No scored tweets found in the results file.")
                    except json.JSONDecodeError:
                        st.error("Could not parse the results file. It may be corrupted.")
                        
                # As a fallback, check for the batch analysis file
                else:
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
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Refresh Now"):
                    st.rerun()
            
            # Auto-refresh every few seconds if enabled
            if auto_refresh:
                time.sleep(3)  # Wait 3 seconds
                st.rerun()

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
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Process tweet with existing semantic analyzer
    analyzer = SimpleSemanticAnalyzer()
    
    # Try to get the campaign brief from the campaign_id
    campaign_brief = ""
    
    # Try multiple possible file locations for the campaign brief
    possible_files = [
        f"./data/{tweet.campaign_id}.json",  # Standard location
        f"./data/{tweet.campaign_id.replace('-', '_')}.json",  # With underscores instead of hyphens
        "./data/test_campaign.json",  # Default test campaign
        "./campaign_brief.txt"  # Raw text file
    ]
    
    print(f"Looking for campaign brief in multiple locations...")
    for file_path in possible_files:
        print(f"Checking {file_path}...")
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.json'):
                    print(f"JSON file found at {file_path}, loading brief...")
                    with open(file_path, "r") as f:
                        campaign_data = json.load(f)
                        if isinstance(campaign_data, dict) and "campaign_brief" in campaign_data:
                            campaign_brief = campaign_data.get("campaign_brief", "")
                            print(f"Loaded campaign brief from {file_path}: {campaign_brief[:100]}...")
                            break
                else:
                    print(f"Text file found at {file_path}, loading brief...")
                    with open(file_path, "r") as f:
                        campaign_brief = f.read()
                        print(f"Loaded campaign brief from {file_path}: {campaign_brief[:100]}...")
                        break
            except Exception as e:
                print(f"Error loading campaign brief from {file_path}: {str(e)}")
    
    # Make sure we have a non-empty brief
    if not campaign_brief or not campaign_brief.strip():
        print("⚠️ Empty campaign brief! Analysis will not work properly.")
        # Use a hardcoded brief as last resort
        campaign_brief = """Pindora LUCIA by Pindora: Your private AI for managing social media, tracking wallets, remembering chats and documents, and completing personal tasks for you 24/7
"""
        print(f"Using hardcoded brief as last resort: {campaign_brief[:100]}...")
    
    # Analyze the tweet with the campaign brief
    score, context = analyzer.analyze_tweet(
        campaign_brief=campaign_brief,
        tweet=tweet.content
    )
    
    # Store the analyzed tweet
    analyzed_tweet = {
        "tweet_id": tweet.tweet_id,
        "content": tweet.content,
        "author_id": tweet.author_id,
        "campaign_id": tweet.campaign_id,
        "scraped_at": tweet.scraped_at,
        "analyzed_at": datetime.now().isoformat(),
        "score": float(score),
        "context": context
    }
    
    # Save to a real-time results file for this campaign
    # Ensure consistent naming format for the results file
    # If the campaign_id already contains the username, use it directly
    # Otherwise, create a standardized format
    if "-" in tweet.campaign_id and tweet.author_id.lower() in tweet.campaign_id.lower():
        # Already using the new format (username-date)
        results_file = f"./data/{tweet.campaign_id}-realtime-results.json"
    else:
        # Using the old format or a custom ID, standardize it
        today = datetime.now().strftime('%Y%m%d')
        standardized_id = f"{tweet.author_id.lower()}-{today}"
        results_file = f"./data/{standardized_id}-realtime-results.json"
        print(f"Standardizing campaign ID from {tweet.campaign_id} to {standardized_id}")
    
    # Load existing results if available
    existing_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []
    
    # Add the new result
    existing_results.append(analyzed_tweet)
    
    # Save the updated results
    with open(results_file, "w") as f:
        json.dump(existing_results, f)
    
    # Also save a copy to the scraper's data directory if needed
    try:
        # Check if the scraper directory exists
        scraper_data_dir = "../scraping-repo/scraper/data"
        if os.path.exists(scraper_data_dir):
            # Create a symlink to the results file in the scraper's data directory
            # This ensures both locations have access to the same data
            scraper_results_file = f"{scraper_data_dir}/{tweet.campaign_id}-realtime-results.json"
            
            # Create the symlink if it doesn't exist
            if not os.path.exists(scraper_results_file):
                # On Windows use: os.symlink(results_file, scraper_results_file)
                # On Unix/Mac:
                os.system(f"ln -sf {os.path.abspath(results_file)} {scraper_results_file}")
                print(f"Created symlink to results in scraper directory: {scraper_results_file}")
    except Exception as e:
        print(f"Note: Could not create symlink to scraper directory: {str(e)}")
    
    print(f"Processed tweet {tweet.tweet_id} with score {score:.2f}/10")
    return {"status": "processed", "score": float(score), "text": tweet.content}

if __name__ == "__main__":
    main()
