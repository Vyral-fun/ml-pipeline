import os
import re
import json
import hashlib
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
import requests
from openai import OpenAI
from dotenv import load_dotenv
import random
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

load_dotenv()

class SemanticAnalyzer:
    def __init__(self, 
                 bi_encoder_model_name: str = "text-embedding-3-small", 
                 cache_dir: str = ".cache",
                 use_rl: bool = True):
        self.bi_encoder_model_name = bi_encoder_model_name
        self.cache_dir = cache_dir
        self.use_rl = use_rl
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.use_twitter_api = self.twitter_bearer_token is not None
        
        if self.use_rl:
            self._initialize_rl_agent()
        else:
            self.rl_agent = None
            print("Reinforcement Learning disabled.")
    
    def _initialize_rl_agent(self):
        try:
            print("Initializing RL agent for scoring optimization...")
            self.rl_agent = PPO("MlpPolicy", DummyVecEnv([lambda: self._create_rl_env()]), verbose=0)
            print("RL agent initialized successfully.")
        except Exception as e:
            print(f"Error initializing RL agent: {e}")
            print("RL functionality will be disabled.")
            self.rl_agent = None
            self.use_rl = False
    
    def _create_rl_env(self):
        from gym import Env
        from gym.spaces import Box, Discrete
        
        class ScoringEnv(Env):
            def __init__(self):
                super(ScoringEnv, self).__init__()
                self.action_space = Box(low=0.5, high=2.0, shape=(1,), dtype=np.float32)  
                self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)  
                self.state = np.array([0.0, 0.0])
                self.step_count = 0
                self.max_steps = 100
            
            def step(self, action):
                self.step_count += 1
                multiplier = action[0]
                similarity = self.state[0]
                score = similarity * multiplier * 10.0
                reward = self._calculate_reward(score)
                self.state = np.array([np.random.random(), np.random.random()])
                done = self.step_count >= self.max_steps
                return self.state, reward, done, {}
            
            def reset(self):
                self.step_count = 0
                self.state = np.array([np.random.random(), np.random.random()])
                return self.state
            
            def _calculate_reward(self, score):
                target_score = 7.0  
                return -abs(score - target_score) / 10.0
        
        return ScoringEnv()
    
    def _get_rl_adjusted_score(self, bi_encoder_score: float, context_features: List[float]) -> float:
        if not self.use_rl or self.rl_agent is None:
            return bi_encoder_score * 10.0
        
        try:
            state = np.array([bi_encoder_score] + context_features[:1])  
            if len(state) < 2:  
                state = np.pad(state, (0, 2 - len(state)), mode='constant')
            action, _ = self.rl_agent.predict(state, deterministic=True)
            multiplier = action[0] if isinstance(action, np.ndarray) else action
            adjusted_score = bi_encoder_score * multiplier * 10.0
            adjusted_score = max(0.0, min(10.0, adjusted_score))
            return adjusted_score
        except Exception as e:
            print(f"Error in RL adjustment: {e}")
            return bi_encoder_score * 10.0
    
    def _update_rl_agent(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        if self.use_rl and self.rl_agent is not None:
            try:
                pass
            except Exception as e:
                print(f"Error updating RL agent: {e}")
    
    def _get_bi_encoder_cache_path(self, text: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{self.bi_encoder_model_name}_{text_hash}.json")
    
    def get_bi_encoder_embedding(self, text: str) -> Optional[List[float]]:
        text = str(text).replace("\n", " ") 
        if not text.strip():  
            return None
        
        cache_path = self._get_bi_encoder_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)['embedding']
            except (json.JSONDecodeError, KeyError, TypeError):
                pass 
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.bi_encoder_model_name,
                input=text
            )
            embedding = response.data[0].embedding
            with open(cache_path, 'w') as f:
                json.dump({'text': text, 'embedding': embedding}, f)
            return embedding
        except Exception as e:
            print(f"Error getting embedding for text '{text[:50]}...': {e}")
            return None
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        if embedding1 is None or embedding2 is None or not embedding1 or not embedding2:
            return 0.0
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def rerank_with_cross_encoder(self, campaign_brief: str, tweet_texts: List[str]) -> List[Optional[float]]:
        print("Cross-encoder functionality has been removed. Using only bi-encoder scores.")
        return [None for _ in tweet_texts]
    
    def analyze_campaign_data(
        self, 
        input_csv_path: str, 
        campaign_brief_path: str,
        output_csv_path: Optional[str] = None,
        batch_size: int = 10,
        top_n_rerank: int = 50
    ) -> pd.DataFrame:
        print(f"Analyzing campaign data from {input_csv_path} using only bi-encoder ({self.bi_encoder_model_name})...")
        if self.use_rl and self.rl_agent is not None:
            print("Reinforcement Learning enabled for score adjustment.")
        
        with open(campaign_brief_path, 'r', encoding='utf-8') as f:
            campaign_brief = f.read().strip()
        print(f"Campaign brief loaded ({len(campaign_brief)} characters). Generating embedding...")
        campaign_embedding = self.get_bi_encoder_embedding(campaign_brief)
        if campaign_embedding is None:
            print("Error: Could not generate embedding for campaign brief. Aborting.")
            return pd.DataFrame()
        
        df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(df)} rows from {input_csv_path}.")
        
        if 'content' not in df.columns:
            print("Error: 'content' column not found in CSV. Available columns:", df.columns)
            return df
        
        relevance_scores = []
        contexts = []
        processed = 0
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_tweets = df['content'][i:batch_end].tolist()
            batch_embeddings = [self.get_bi_encoder_embedding(t) for t in batch_tweets]
            
            for j, emb in enumerate(batch_embeddings):
                if emb is not None and campaign_embedding is not None:
                    sim = self.calculate_cosine_similarity(campaign_embedding, emb)
                    score = self._get_rl_adjusted_score(sim, [len(batch_tweets[j]) / 280.0])  
                    context = f"Bi-encoder ({self.bi_encoder_model_name}) cosine similarity: {sim:.3f}"
                    if self.use_rl and self.rl_agent is not None:
                        context += "; RL adjusted"
                else:
                    score = 0.0
                    context = "Embedding failed"
                relevance_scores.append(score)
                contexts.append(context)
                processed += 1
                if processed % 10 == 0 or processed == len(df):
                    print(f"Processed {processed}/{len(df)} tweets...")
        
        df['relevance_score'] = relevance_scores
        df['relevance_context'] = contexts
        
        print(f"Analysis complete. Relevance scores added to {len(df)} rows using only bi-encoder.")
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")
        return df
    
    def test_with_sample_content(self, campaign_brief_path: str, test_tweets: List[str], top_n_rerank: int = 5) -> None:
        print("Testing semantic analyzer with sample content using only bi-encoder...")
        if self.use_rl and self.rl_agent is not None:
            print("Reinforcement Learning enabled for score adjustment.")
        
        with open(campaign_brief_path, 'r', encoding='utf-8') as f:
            campaign_brief = f.read().strip()
        print(f"Campaign brief loaded ({len(campaign_brief)} characters). Generating embedding...")
        campaign_embedding = self.get_bi_encoder_embedding(campaign_brief)
        if campaign_embedding is None:
            print("Error: Could not generate embedding for campaign brief. Aborting.")
            return
        
        print(f"Analyzing {len(test_tweets)} sample tweets with bi-encoder ({self.bi_encoder_model_name})...")
        bi_encoder_results = []
        for i, tweet in enumerate(test_tweets):
            tweet_embedding = self.get_bi_encoder_embedding(tweet)
            if tweet_embedding is not None and campaign_embedding is not None:
                sim = self.calculate_cosine_similarity(campaign_embedding, tweet_embedding)
                score = self._get_rl_adjusted_score(sim, [len(tweet) / 280.0])  
                context = f"Bi-encoder ({self.bi_encoder_model_name}) cosine similarity: {sim:.3f}"
                if self.use_rl and self.rl_agent is not None:
                    context += "; RL adjusted"
            else:
                score = 0.0
                context = "Embedding failed"
            bi_encoder_results.append({
                'original_index': i,
                'text': tweet,
                'bi_encoder_score': score,
                'context': context
            })
        
        bi_encoder_results.sort(key=lambda x: x['bi_encoder_score'], reverse=True)
        print("\nResults (sorted by bi-encoder score, highest to lowest):")
        final_results_map = {i: {'final_score': 0.0, 'context': ''} for i in range(len(test_tweets))}
        for i, res in enumerate(bi_encoder_results):
            original_idx = res['original_index']
            final_results_map[original_idx]['final_score'] = res['bi_encoder_score']
            final_results_map[original_idx]['context'] = res['context']
        
        for i in range(len(test_tweets)):
            result = final_results_map[i]
            print(f"Sample Tweet #{i+1}: {result['text'][:100]}...")
            print(f"  Final Score: {result['final_score']:.2f}/10 ({result['context']})")    

def main():
    parser = argparse.ArgumentParser(description='Semantic Analyzer (Bi-Encoder Only with RL)')
    parser.add_argument('--csv', type=str, default="CoralApp is Coming-campaign-report-07-05-2025.csv")
    parser.add_argument('--brief', type=str, default="campaign_brief.txt")
    parser.add_argument('--output', type=str, default="CoralApp-campaign-report-with-relevance.csv")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--bi-model', type=str, default="text-embedding-3-small", help='Bi-encoder model (OpenAI ID)')
    parser.add_argument('--top-n-rerank', type=int, default=50, help='Ignored, as cross-encoder is removed')
    parser.add_argument('--test', action='store_true', help='Run with sample tweets')
    parser.add_argument('--no-rl', action='store_true', help='Disable reinforcement learning for score adjustment')
    args = parser.parse_args()

    print("Starting Semantic Analyzer (Bi-Encoder Only with RL)...")
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY for bi-encoder not set.")
        return

    analyzer = SemanticAnalyzer(
        bi_encoder_model_name=args.bi_model,
        use_rl=not args.no_rl
    )

    if args.test:
        sample_tweets = [
            "Just downloaded #CoralApp and I'm loving how it connects me with people who share my interests! The UI is so intuitive.",
            "This new social media app I found has the best communities! #CoralApp #ConnectDeeper",
            "Why is my phone so slow today? Need to clear some space and delete unused apps.",
            "Check out these amazing photos I took on my vacation to Hawaii last week!",
            "CoralApp's privacy settings are exactly what I've been looking for in a social media platform. Finally feeling in control of my data!",
            "Exploring the features of CoralApp. The interest-based communities are a game changer. Highly recommend for meaningful connections.",
            "Is anyone else having trouble with their internet connection today? It's so frustrating.",
            "Can CoralApp compete with established giants? Its focus on privacy and curated content is a strong differentiator. #SocialMediaFuture",
            "I'm so excited to see the new features of CoralApp! The community is amazing and the content is so relevant to my interests. #SocialMedia",
            "CoralApp"
        ]
        analyzer.test_with_sample_content(args.brief, sample_tweets, top_n_rerank=args.top_n_rerank if args.top_n_rerank > 0 else len(sample_tweets))
    else:
        analyzer.analyze_campaign_data(
            args.csv, 
            args.brief, 
            args.output, 
            args.batch_size,
            args.top_n_rerank
        )

if __name__ == "__main__":
    main()