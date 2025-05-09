# Deploying to Streamlit Cloud

This guide walks you through the process of deploying the Yap.market app to Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit Cloud account (sign up at https://streamlit.io/cloud)
- OpenAI API key

## Step 1: Prepare your repository

1. Initialize git repository (if not already done):
   ```bash
   cd /Users/ayush/Documents/social-listening/yap-market
   git init
   ```

2. Add all your files to git:
   ```bash
   git add .
   ```

3. Commit your changes:
   ```bash
   git commit -m "Initial commit"
   ```

## Step 2: Create GitHub repository

1. Go to GitHub.com and create a new repository
2. Follow the instructions to push your existing repository:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/yap-market.git
   git branch -M main
   git push -u origin main
   ```

## Step 3: Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your repository, branch, and `app.py` as the main file
4. Under "Advanced settings":
   - Add your OpenAI API key as a secret:
     - Key: `OPENAI_API_KEY`
     - Value: `your_api_key_here`
   - (Optional) Set Python version to 3.10
5. Click "Deploy"

## Step 4: Troubleshooting

If you encounter any issues with the deploy:

1. Check the logs in Streamlit Cloud
2. Verify your `requirements.txt` file contains all necessary dependencies
3. Make sure your API key is correctly set in the Streamlit Cloud secrets

## Additional Notes

- Your app will be available at a URL like: `https://username-appname-randomstring.streamlit.app/`
- Any updates you push to your GitHub repository will automatically trigger a redeploy

## Managing Your App

- You can manage, stop, or restart your app from the Streamlit Cloud dashboard
- Monitor usage statistics to see how your app is performing
