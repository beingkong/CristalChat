"""
Speech Emotion Recognition (SER) module for detecting emotions in speech.
"""

import numpy as np
from typing import Dict, List, Optional, Union

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False


class SERModule:
    """
    Speech Emotion Recognition module that detects emotions in speech.
    This implementation leverages SenseVoice's built-in emotion detection capabilities.
    """
    
    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize the SER module.
        
        Args:
            model_path (str, optional): Path to emotion model. If None, will use SenseVoice's built-in capabilities.
            device (str): Device to run the model on ("cuda" or "cpu")
        """
        # Check if CUDA is available and fall back to CPU if not
        if device == "cuda" and (not TORCH_AVAILABLE or (TORCH_AVAILABLE and not torch.cuda.is_available())):
            print("SER: CUDA not available, falling back to CPU")
            device = "cpu"
            
        self.device = device
        self.model = None
        self.audio_buffer = []
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        
        # In a real implementation, we might load a separate emotion model
        # or just rely on SenseVoice's built-in emotion detection
        self._load_model(model_path)
        print(f"SER: Initialized on {device}")
    
    def _load_model(self, model_path):
        """
        Load the emotion recognition model.
        
        In a real implementation with a separate model, this would load it.
        For this implementation, we'll simulate the model.
        """
        # Simulate model loading
        print("SER: Using SenseVoice's built-in emotion detection (simulated)")
        # For simulation purposes
        self.model = SimulatedEmotionModel()
    
    def start_detection(self):
        """Reset the SER state for a new detection session."""
        self.audio_buffer = []
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        print("SER: Started new emotion detection session")
    
    def consume_audio(self, audio_frame):
        """
        Process an audio frame for emotion detection.
        
        Args:
            audio_frame (numpy.ndarray): Audio frame data
            
        Returns:
            None
        """
        if audio_frame is None:
            return
        
        # Add frame to buffer
        self.audio_buffer.append(audio_frame)
        
        # Process every 20 frames (more frames needed for reliable emotion detection)
        if len(self.audio_buffer) >= 20:
            self._process_buffer()
    
    def _process_buffer(self):
        """
        Process the current audio buffer for emotion detection.
        """
        # Concatenate audio frames
        audio = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([])
        
        if len(audio) == 0:
            return
        
        # In a real implementation, this would call the emotion model
        # result = self.model.detect_emotion(audio)
        
        # For simulation, we'll use our simulated model
        result = self.model.detect_emotion(audio)
        
        # Extract results
        emotion = result.get("emotion", "neutral")
        confidence = result.get("confidence", 0.0)
        
        # Update emotion if confidence is higher than current
        if confidence > self.emotion_confidence:
            self.current_emotion = emotion
            self.emotion_confidence = confidence
        
        # Keep a sliding window of audio for continuous emotion detection
        # Only keep the most recent half of the buffer
        half_size = len(self.audio_buffer) // 2
        self.audio_buffer = self.audio_buffer[half_size:]
    
    def get_detected_emotion(self):
        """
        Get the detected emotion.
        
        Returns:
            str: Detected emotion (e.g., "happy", "sad", "angry", "neutral")
        """
        # Process any remaining audio
        if len(self.audio_buffer) > 5:  # Only process if we have enough frames
            self._process_buffer()
        
        return self.current_emotion
    
    def get_emotion_confidence(self):
        """
        Get the confidence score for the detected emotion.
        
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        return self.emotion_confidence


class SimulatedEmotionModel:
    """
    A simulated emotion detection model for demonstration purposes.
    In a real implementation, this would be replaced with an actual emotion model
    or would use SenseVoice's built-in emotion detection.
    """
    
    def __init__(self):
        """Initialize the simulated model."""
        self.emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    
    def detect_emotion(self, audio):
        """
        Simulate emotion detection by returning a random emotion.
        
        Args:
            audio (numpy.ndarray): Audio data
            
        Returns:
            dict: Simulated emotion detection result
        """
        # For simulation, we'll return a random emotion with random confidence
        # In a real implementation, this would analyze the audio and return actual emotion
        import random
        
        emotion = random.choice(self.emotions)
        confidence = random.uniform(0.6, 0.95)
        
        return {
            "emotion": emotion,
            "confidence": confidence
        }
