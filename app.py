import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
from dotenv import load_dotenv
# No need for sentence_transformers import - using only OpenAI embeddings

# Load environment variables
load_dotenv()

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

st.set_page_config(
    page_title="Yap.market Social Listening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Make all text white by default */
    .stApp {
        color: white !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    /* Ensure sidebar headers are white */
    .sidebar .sub-header {
        color: white;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        color: white;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
    .score-high {
        color: #4caf50;
        font-weight: bold;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.5);
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.5);
    }
    .score-low {
        color: #f44336;
        font-weight: bold;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.5);
    }
    /* Force white text in all Streamlit elements */
    .stMarkdown, .stDataFrame, .stText, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: white !important;
    }
    div.element-container div.stMarkdown, div.element-container div.stDataFrame {
        color: white !important;
    }
    /* Make metric value text white */
    .stMetric .stMetricValue {
        color: white !important;
    }
    /* Make all input fields and selects have white text */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        color: white !important;
    }
    /* Make dataframe text white */
    .dataframe {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def format_score(score):
    """Format score with appropriate color class based on value"""
    if score >= 7.0:
        return f'<span class="score-high">{score:.2f}</span>'
    elif score >= 4.0:
        return f'<span class="score-medium">{score:.2f}</span>'
    else:
        return f'<span class="score-low">{score:.2f}</span>'

def main():
    # Header
    st.markdown('<h1 class="main-header">Yap.market Social Listening</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        <p>This tool analyzes social media content for relevance to marketing campaigns using AI. 
        It measures how well tweets match campaign objectives and provides relevance scores (0-10).</p>
        <p>The system uses OpenAI embeddings with optional reinforcement learning to improve scoring over time.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", 
                                  help="Enter your OpenAI API key or set it in .env file")
    
    # OpenAI embeddings model selection
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=1,
        help="Select OpenAI embedding model to use for similarity calculation"
    )
    
    # Model info
    st.sidebar.markdown(f"**Using model:** {embedding_model}")
    st.sidebar.markdown("*Using OpenAI embeddings for semantic similarity scoring*")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This app demonstrates semantic analysis for social listening using OpenAI embeddings. "
        "It measures how well tweets match campaign objectives."
    )
    # No fine-tuning model selection needed
    
    # Create tabs
    tab1, tab2 = st.tabs(["Campaign Analysis", "Batch Processing"])
    
    # No fine-tuning tab content needed
    
    # Tab 1: Campaign Analysis
    with tab1:
        st.markdown('<h2 class="sub-header">Campaign Analysis</h2>', unsafe_allow_html=True)
        
        # Campaign brief input
        st.markdown("### Campaign Brief")
        campaign_brief = st.text_area(
            "Enter campaign brief details:",
            height=150,
            placeholder="Describe the campaign objectives, target audience, key messages, etc."
        )
        
        # Tweet analysis
        st.markdown("### Tweet Analysis")
        
        # Dynamic tweet input
        tweet_inputs = []
        col1, col2 = st.columns([3, 1])
        with col1:
            num_tweets = st.number_input("Number of tweets to analyze:", min_value=1, max_value=10, value=3)
        
        for i in range(int(num_tweets)):
            tweet = st.text_area(f"Tweet {i+1}:", height=100, key=f"tweet_{i}")
            tweet_inputs.append(tweet)
        
        # Analyze button
        if st.button("Analyze Tweets"):
            if not campaign_brief:
                st.error("Please enter campaign brief details.")
            elif not any(tweet.strip() for tweet in tweet_inputs):
                st.error("Please enter at least one tweet to analyze.")
            else:
                try:
                    # Initialize analyzer
                    analyzer = SimpleSemanticAnalyzer(
                        api_key=api_key
                    )
                    
                    with st.spinner("Analyzing tweets..."):
                        # Filter out empty tweets
                        valid_tweets = [t for t in tweet_inputs if t.strip()]
                        
                        # Analyze tweets
                        results_df = analyzer.analyze_tweets(campaign_brief, valid_tweets)
                        
                        # Display results
                        # results_df is already a DataFrame from analyze_tweets
                        
                        # Sort by relevance score
                        results_df = results_df.sort_values(by="Relevance Score", ascending=False)
                        
                        # Display results in a nice format
                        st.markdown("### Analysis Results")
                        for i, row in results_df.iterrows():
                            st.markdown(
                                f"""
                                <div class="card">
                                    <h4 style="color: white;">Tweet {i+1}</h4>
                                    <p style="color: white;">{row['Tweet']}</p>
                                    <p style="color: white;"><strong>Relevance Score:</strong> {format_score(row['Relevance Score'])}/10</p>
                                    <p style="color: white;"><small>{row['Context']}</small></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Visualization
                        st.markdown("### Visualization")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Create color map based on scores
                        colors = ['#c62828', '#f9a825', '#2e7d32']
                        cmap = sns.blend_palette(colors, as_cmap=True)
                        
                        # Create horizontal bar chart
                        bars = ax.barh(
                            y=range(len(results_df)),
                            width=results_df['Relevance Score'],
                            color=[cmap(score/10) for score in results_df['Relevance Score']]
                        )
                        
                        # Add tweet text as y-tick labels (truncated)
                        ax.set_yticks(range(len(results_df)))
                        ax.set_yticklabels([text[:50] + '...' if len(text) > 50 else text for text in results_df['Tweet']])
                        
                        # Add score values at the end of each bar
                        for i, bar in enumerate(bars):
                            ax.text(
                                bar.get_width() + 0.2,
                                bar.get_y() + bar.get_height()/2,
                                f"{results_df['Relevance Score'].iloc[i]:.2f}",
                                va='center'
                            )
                        
                        # Set chart title and labels
                        ax.set_title('Tweet Relevance Scores')
                        ax.set_xlabel('Relevance Score (0-10)')
                        ax.set_xlim(0, 11)  # Set x-axis limit to accommodate score labels
                        
                        # Display the chart
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error analyzing tweets: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if 'campaign_brief_path' in locals():
                        os.unlink(campaign_brief_path)
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Processing</h2>', unsafe_allow_html=True)
        st.markdown("Upload a CSV file with tweets and a campaign brief file to analyze in batch.")
        
        # File uploads
        uploaded_csv = st.file_uploader("Upload CSV with tweets (must have 'content' column):", type=["csv"])
        uploaded_brief = st.file_uploader("Upload campaign brief:", type=["txt"])
        
        if st.button("Process Batch") and uploaded_csv is not None and uploaded_brief is not None:
            try:
                # Read CSV and brief
                df = pd.read_csv(uploaded_csv)
                campaign_brief = uploaded_brief.getvalue().decode('utf-8')
                
                if 'content' not in df.columns:
                    st.error("CSV must contain a 'content' column with tweet text.")
                else:
                    # Initialize analyzer
                    analyzer = SimpleSemanticAnalyzer(
                        api_key=api_key
                    )
                    
                    with st.spinner("Processing batch..."):
                        # Process tweets in batches
                        tweets = df['content'].tolist()
                        results_df = analyzer.analyze_tweets(campaign_brief, tweets)
                        
                        # Merge results with original dataframe
                        output_df = pd.concat([df, results_df.drop('Tweet', axis=1)], axis=1)
                    
                    # Display results
                    st.markdown("### Batch Results")
                    st.dataframe(output_df)
                    
                    # Option to download results
                    csv = output_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="campaign_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("### Summary Statistics")
                    avg_score = output_df['relevance_score'].mean()
                    median_score = output_df['relevance_score'].median()
                    high_relevance = (output_df['relevance_score'] >= 7.0).sum()
                    medium_relevance = ((output_df['relevance_score'] >= 4.0) & (output_df['relevance_score'] < 7.0)).sum()
                    low_relevance = (output_df['relevance_score'] < 4.0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{avg_score:.2f}")
                    with col2:
                        st.metric("Median Score", f"{median_score:.2f}")
                    with col3:
                        st.metric("Total Tweets", len(output_df))
                    
                    # Distribution chart
                    st.markdown("### Score Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create histogram with KDE
                    sns.histplot(output_df['relevance_score'], kde=True, ax=ax)
                    
                    # Add vertical lines for score categories
                    ax.axvline(x=4.0, color='orange', linestyle='--', alpha=0.7)
                    ax.axvline(x=7.0, color='green', linestyle='--', alpha=0.7)
                    
                    # Add text annotations
                    ax.text(2.0, ax.get_ylim()[1]*0.9, f"Low: {low_relevance}", color='#c62828', fontweight='bold')
                    ax.text(5.5, ax.get_ylim()[1]*0.9, f"Medium: {medium_relevance}", color='#f9a825', fontweight='bold')
                    ax.text(8.5, ax.get_ylim()[1]*0.9, f"High: {high_relevance}", color='#2e7d32', fontweight='bold')
                    
                    ax.set_title('Distribution of Relevance Scores')
                    ax.set_xlabel('Relevance Score (0-10)')
                    ax.set_ylabel('Count')
                    
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
            
            finally:
                # Clean up temporary files
                if 'csv_path' in locals():
                    os.unlink(csv_path)
                if 'brief_path' in locals():
                    os.unlink(brief_path)

if __name__ == "__main__":
    main()
