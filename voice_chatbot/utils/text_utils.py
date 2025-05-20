"""
Text utility functions for voice chatbot.
"""

import re


def clean_text(text):
    """
    Clean and normalize text for TTS processing.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    # Remove any special markdown or formatting characters
    text = re.sub(r'[*_~`#]', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'['']', "'", text)
    
    # Ensure text ends with punctuation for better TTS prosody
    if text and not re.search(r'[.,!?;:]$', text):
        text = text + '.'
        
    return text.strip()


def compose_prompt(user_text, emotion, context_history):
    """
    Compose a prompt for the LLM including user text, emotion, and context.
    
    Args:
        user_text (str): User's transcribed text
        emotion (str): Detected emotion (e.g., "happy", "sad", "angry", "neutral")
        context_history (list): List of previous conversation turns
        
    Returns:
        str: Formatted prompt for LLM
    """
    # Format context history
    history_text = ""
    if context_history:
        for i, entry in enumerate(context_history):
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role and content:
                history_text += f"{role.capitalize()}: {content}\n"
    
    # Create emotion context
    emotion_context = ""
    if emotion and emotion.lower() != "neutral":
        emotion_context = f"[The user sounds {emotion}. Respond appropriately to their emotional state.]"
    
    # Combine all elements
    prompt = f"{history_text}\nUser: {user_text}\n{emotion_context}\nAssistant:"
    
    return prompt


def save_to_history(role, content):
    """
    Create a history entry for conversation context.
    
    Args:
        role (str): The speaker role ("user" or "assistant")
        content (str): The message content
        
    Returns:
        dict: A history entry
    """
    return {
        "role": role,
        "content": content
    }
