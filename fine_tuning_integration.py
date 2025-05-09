"""
Integration module to connect fine-tuned models with Streamlit app
"""

import os
import json
import streamlit as st
from fine_tuning import CrossEncoderFineTuner
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTuningManager:
    """
    Manager class for fine-tuning operations within the Streamlit app
    """
    
    def __init__(self, models_dir: str = "fine_tuned_models"):
        """
        Initialize the fine-tuning manager
        
        Args:
            models_dir: Directory where fine-tuned models are saved
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing models info
        self.models_info_path = os.path.join(models_dir, "models_info.json")
        self.models_info = self._load_models_info()
        
    def _load_models_info(self) -> Dict:
        """
        Load information about fine-tuned models
        
        Returns:
            Dictionary with model information
        """
        if os.path.exists(self.models_info_path):
            try:
                with open(self.models_info_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse models info file: {self.models_info_path}")
                return {"models": []}
        else:
            return {"models": []}
    
    def _save_models_info(self):
        """Save current models info to disk"""
        with open(self.models_info_path, 'w') as f:
            json.dump(self.models_info, f, indent=2)
    
    def get_available_models(self) -> List[Dict]:
        """
        Get list of available fine-tuned models
        
        Returns:
            List of dictionaries with model information
        """
        return self.models_info.get("models", [])
    
    def initiate_fine_tuning(
        self,
        csv_path: str,
        campaign_brief_path: str,
        model_name: str,
        base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        label_column: Optional[str] = None,
        content_column: str = "content",
        epochs: int = 3,
        use_openai: bool = False,
        batch_size: int = 16
    ) -> str:
        """
        Start fine-tuning process
        
        Args:
            csv_path: Path to CSV with training data
            campaign_brief_path: Path to campaign brief
            model_name: Name for the fine-tuned model
            base_model: Base cross-encoder model
            label_column: Column with relevance scores
            content_column: Column with tweet content
            epochs: Number of training epochs
            use_openai: Whether to use OpenAI for scoring
            batch_size: Training batch size
            
        Returns:
            Status message
        """
        # Create a unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name.replace(' ', '_')}_{timestamp}"
        output_dir = os.path.join(self.models_dir, model_id)
        
        # Initialize fine-tuner
        fine_tuner = CrossEncoderFineTuner(
            base_model=base_model,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        try:
            # Prepare dataset
            train_data, val_data = fine_tuner.prepare_dataset_from_csv(
                csv_path=csv_path,
                campaign_brief_path=campaign_brief_path,
                label_column=label_column,
                content_column=content_column,
                use_openai_scoring=use_openai
            )
            
            # Display dataset info
            st.write(f"Prepared dataset with {len(train_data)} training and {len(val_data)} validation examples")
            
            # Fine-tune the model
            model_path = fine_tuner.fine_tune(train_data, val_data)
            
            # Update models info
            model_info = {
                "id": model_id,
                "name": model_name,
                "base_model": base_model,
                "created_at": timestamp,
                "path": model_path,
                "campaign_brief": campaign_brief_path,
                "training_data": csv_path,
                "epochs": epochs,
                "train_examples": len(train_data),
                "val_examples": len(val_data)
            }
            
            self.models_info["models"].append(model_info)
            self._save_models_info()
            
            return f"Fine-tuning completed successfully. Model saved as '{model_name}'."
        
        except Exception as e:
            logger.error(f"Fine-tuning error: {str(e)}")
            return f"Error during fine-tuning: {str(e)}"
    
    def render_fine_tuning_ui(self):
        """Render Streamlit UI for fine-tuning operations"""
        st.markdown("## Model Fine-Tuning")
        
        with st.expander("Fine-Tune New Model", expanded=False):
            st.write("Fine-tune a cross-encoder model on your campaign data to improve relevance scoring.")
            
            # Form for fine-tuning configuration
            with st.form("fine_tuning_form"):
                model_name = st.text_input("Model Name", value="Campaign Relevance Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    base_model = st.selectbox(
                        "Base Model",
                        [
                            "cross-encoder/ms-marco-MiniLM-L-6-v2",
                            "cross-encoder/stsb-roberta-base",
                            "cross-encoder/ms-marco-MiniLM-L-12-v2"
                        ]
                    )
                    epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=3)
                
                with col2:
                    batch_size = st.slider("Batch Size", min_value=4, max_value=32, value=16, step=4)
                    use_openai = st.checkbox("Use OpenAI for Labeling", value=False)
                
                # File upload for training data
                uploaded_csv = st.file_uploader("Upload Training CSV", type=["csv"])
                uploaded_brief = st.file_uploader("Upload Campaign Brief", type=["txt"])
                
                # Advanced options (using a section with a divider instead of an expander)
                st.markdown("---")
                st.markdown("**Advanced Options:**")
                
                # Display CSV columns if a file is uploaded
                if uploaded_csv is not None:
                    try:
                        # Read CSV to get column names
                        csv_df = pd.read_csv(uploaded_csv)
                        available_columns = csv_df.columns.tolist()
                        
                        # Show column selector with available options
                        content_column = st.selectbox(
                            "Content Column (Tweet Text)", 
                            options=available_columns,
                            index=0 if available_columns else None,
                            help="Select the column containing the tweet text"
                        )
                        
                        # Show potential score columns
                        score_columns = [col for col in available_columns if any(term in col.lower() for term in ['score', 'rating', 'relevance', 'rank'])]
                        label_column = st.selectbox(
                            "Label Column (Relevance Score)", 
                            options=["None (Use OpenAI scoring)"] + available_columns,
                            index=0,
                            help="Select the column containing relevance scores, or use OpenAI to generate scores"
                        )
                        if label_column == "None (Use OpenAI scoring)":
                            label_column = ""
                            
                        # Show sample data
                        with st.expander("Preview CSV Data"):
                            st.dataframe(csv_df.head(3))
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
                        content_column = st.text_input("Content Column", value="content")
                        label_column = st.text_input("Label Column (leave empty to use OpenAI scoring)", value="")
                else:
                    content_column = st.text_input("Content Column", value="content")
                    label_column = st.text_input("Label Column (leave empty to use OpenAI scoring)", value="")
                
                submit_button = st.form_submit_button("Start Fine-Tuning")
            
            # Handle form submission
            if submit_button:
                if uploaded_csv is None or uploaded_brief is None:
                    st.error("Please upload both CSV data and campaign brief.")
                else:
                    # Save uploaded files to temporary locations
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
                        tmp_csv.write(uploaded_csv.getvalue())
                        csv_path = tmp_csv.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_brief:
                        tmp_brief.write(uploaded_brief.getvalue())
                        brief_path = tmp_brief.name
                    
                    # Start fine-tuning
                    with st.spinner("Fine-tuning in progress..."):
                        result = self.initiate_fine_tuning(
                            csv_path=csv_path,
                            campaign_brief_path=brief_path,
                            model_name=model_name,
                            base_model=base_model,
                            label_column=label_column if label_column else None,
                            content_column=content_column,
                            epochs=epochs,
                            use_openai=use_openai,
                            batch_size=batch_size
                        )
                    
                    st.success(result)
        
        # Display available models
        st.markdown("## Available Fine-Tuned Models")
        
        models = self.get_available_models()
        if not models:
            st.info("No fine-tuned models available yet. Use the form above to train a new model.")
        else:
            for model in models:
                with st.expander(f"{model['name']} ({model['created_at']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Base Model:** {model['base_model']}")
                        st.write(f"**Training Examples:** {model.get('train_examples', 'N/A')}")
                        st.write(f"**Validation Examples:** {model.get('val_examples', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Epochs:** {model.get('epochs', 'N/A')}")
                        st.write(f"**ID:** {model['id']}")
                        
                        # Option to use this model
                        st.button(
                            "Use This Model",
                            key=f"use_model_{model['id']}",
                            help="Use this model for relevance scoring in the app",
                            # This button just sets a session state variable that the main app will check
                            on_click=lambda mid=model['id']: setattr(st.session_state, 'selected_fine_tuned_model', mid)
                        )
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """
        Get path to a fine-tuned model by ID
        
        Args:
            model_id: ID of the fine-tuned model
            
        Returns:
            Path to the model or None if not found
        """
        for model in self.models_info.get("models", []):
            if model["id"] == model_id:
                return model["path"]
        
        return None

# Example of how to use this in Streamlit app
def add_fine_tuning_tab():
    manager = FineTuningManager()
    manager.render_fine_tuning_ui()
    
    # Return the manager for use in other parts of the app
    return manager
