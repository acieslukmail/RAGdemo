import os
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_keys():
    """
    Check if API keys are available in environment variables
    
    Returns:
        Dictionary with status of each API key
    """
    keys = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", "")
    }
    
    return {
        "openai_status": bool(keys["openai"]),
        "deepseek_status": bool(keys["deepseek"]),
        "openai_key": keys["openai"],
        "deepseek_key": keys["deepseek"]
    }

def display_error(message):
    """
    Display error message in Streamlit
    
    Args:
        message: Error message to display
    """
    st.error(f"❌ {message}")
    logger.error(message)

def display_success(message):
    """
    Display success message in Streamlit
    
    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")
    logger.info(message)

def format_context(context_list, max_length=150):
    """
    Format context fragments for display
    
    Args:
        context_list: List of context text fragments
        max_length: Maximum length for preview
        
    Returns:
        Formatted context HTML
    """
    result = []
    for i, ctx in enumerate(context_list, 1):
        preview = ctx[:max_length] + "..." if len(ctx) > max_length else ctx
        result.append(f"**{i}.** {preview}")
    
    return result