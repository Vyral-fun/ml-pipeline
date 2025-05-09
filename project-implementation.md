Yap.market – Social Listening 
Overview
The Social Listening System will monitor and analyze yapper behavior on Twitter after they join a campaign. It tracks two primary dimensions:
The content the yapper creates


The interactions the yapper performs on the brand’s social content


The system will use semantic analysis (via embeddings and optionally RAG) to evaluate relevance, and build a structured data pipeline that feeds internal scoring systems.

Key Objectives
Track original content and its engagement from the followers for yapper


Track engagements made by the yapper on brand content


Use AI/ML to measure content relevance to campaign objectives


Store all activity and content analysis results in a clean, scalable pipeline



1. Track Yapper-Created Content
Monitor every piece of original Twitter content the yapper posts after joining a campaign.
Actions to capture:
Original tweets


Quote tweets


Replies ( Comments ) 


Retweets


For each post, also collect:
Who liked, retweeted, and commented on it (engagement actors)


Impression count for all the yapper activity 


Timestamp and content



2. Track Yapper’s Brand-Side Activity
Monitor how the yapper engages with the brand’s Twitter content after joining a campaign.
Actions to capture:
Likes on any tweet from the brand


Comments (replies) on brand tweets


Retweets or quote tweets of brand content


This allows us to understand how actively the yapper is supporting the brand even beyond their original content.

3. Semantic Relevance Analysis
Inputs:
campaign_brief.txt provided at campaign creation


Optional contextual data:


Brand’s pinned tweet


Brand bio


Campaign-linked documents or URLs


Processing:
Use embedding models (e.g., OpenAI / OpenKaito) to:


Embed campaign brief and other context


Embed each yapper tweet or comment


Compare using cosine similarity


Optionally use RAG / GAN to:


Retrieve campaign-relevant context


Pass tweet + context to an LLM (e.g., GPT) for a deeper semantic match


Output (per tweet/comment):
Relevance score (0  - 10)


Context used for evaluation (Project content matching details after inputting in RAG) 


Optional LLM-generated explanation (why the content is or isn’t relevant)



4. Data Infrastructure
The system should store everything in a structured and queryable format for internal scoring and analytics.
For yapper-created content:
Tweet ID, author ID, campaign ID


Engagement metrics (likes, RTs, replies, impressions, QT)


List of users who engaged


Semantic Relevance score


Timestamp


For yapper’s brand-side activity:
Brand tweet ID


Yapper twitter ID


Action type (like, comment, RT, QT)


Timestamp


Outputs:
Internal JSON API


CSV export with grouping by:


Campaign


Yapper


Action type


Summary
Track both content created by yappers and their engagements with brand content


Use ML/NLP to measure semantic relevance of each action


Feed structured data into a pipeline that supports campaign reporting and performance-based payouts



