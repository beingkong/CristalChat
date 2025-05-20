"""
Automatic Speech Recognition (ASR) module using SenseVoice model.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False


class ASRModule:
    """
    Automatic Speech Recognition module that uses SenseVoice model
    to convert speech to text with emotion detection.
    """
    
    def __init__(self, model_path=None, device="cuda", use_streaming=True):
        """
        Initialize the ASR module with SenseVoice model.
        
        Args:
            model_path (str, optional): Path to SenseVoice model. If None, will download from HuggingFace.
            device (str): Device to run the model on ("cuda" or "cpu")
            use_streaming (bool): Whether to use streaming mode for incremental recognition
        """
        # Check if CUDA is available and fall back to CPU if not
        if device == "cuda" and (not TORCH_AVAILABLE or (TORCH_AVAILABLE and not torch.cuda.is_available())):
            print("ASR: CUDA not available, falling back to CPU")
            device = "cpu"
            
        self.device = device
        self.use_streaming = use_streaming
        self.model = None
        self.processor = None
        self.audio_buffer = []
        self.cached_features = None
        self.partial_text = ""
        self.detected_language = "en"
        self.detected_emotion = "neutral"
        
        # Load model (in a real implementation, this would use the actual SenseVoice API)
        self._load_model(model_path)
        print(f"ASR: Initialized with SenseVoice model on {device}")
    
    def _load_model(self, model_path):
        """
        Load the SenseVoice model.
        
        In a real implementation, this would use:
        from funasr import AutoModel
        self.model = AutoModel(model_path or "iic/senseVoice-small", device=self.device)
        
        For this implementation, we'll simulate the model loading.
        """
        # Simulate model loading
        print("ASR: Loading SenseVoice model (simulated)")
        # In a real implementation, this would load the actual model
        # self.model = AutoModel(model_path or "iic/senseVoice-small", device=self.device)
        
        # For simulation purposes
        self.model = SimulatedSenseVoiceModel()
        self.processor = self.model.processor
    
    def start_recognition(self):
        """Reset the ASR state for a new recognition session."""
        self.audio_buffer = []
        self.cached_features = None
        self.partial_text = ""
        self.detected_language = "en"
        self.detected_emotion = "neutral"
        print("ASR: Started new recognition session")
    
    def consume_audio(self, audio_frame):
        """
        Process an audio frame and update recognition results.
        
        Args:
            audio_frame (numpy.ndarray): Audio frame data
            
        Returns:
            str or None: Partial recognition result if available, None otherwise
        """
        if audio_frame is None:
            return None
        
        # Add frame to buffer
        self.audio_buffer.append(audio_frame)
        
        # In streaming mode, process incrementally
        if self.use_streaming and len(self.audio_buffer) >= 10:  # Process every 10 frames
            return self._process_buffer(is_final=False)
        
        return None
    
    def _process_buffer(self, is_final=False):
        """
        Process the current audio buffer.
        
        Args:
            is_final (bool): Whether this is the final processing for the utterance
            
        Returns:
            str: Recognition result
        """
        # Concatenate audio frames
        audio = np.concatenate(self.audio_buffer) if self.audio_buffer else np.array([])
        
        if len(audio) == 0:
            return self.partial_text
        
        # In a real implementation, this would call the SenseVoice model
        # result = self.model.transcribe(audio, cache=self.cached_features if self.use_streaming else None)
        # self.cached_features = result.get("cache") if self.use_streaming else None
        
        # For simulation, we'll use our simulated model
        result = self.model.transcribe(audio, is_final=is_final)
        
        # Extract results
        text = result.get("text", "")
        self.detected_language = result.get("language", "en")
        self.detected_emotion = result.get("emotion", "neutral")
        
        # Update partial text
        if is_final:
            self.partial_text = text
            self.audio_buffer = []  # Clear buffer after final processing
        else:
            # For streaming, we might get partial results
            if text and len(text) > len(self.partial_text):
                self.partial_text = text
        
        return self.partial_text
    
    def get_result_text(self):
        """
        Get the final recognition result.
        
        Returns:
            str: Final recognition text
        """
        # Process any remaining audio
        if self.audio_buffer:
            self._process_buffer(is_final=True)
        
        return self.partial_text
    
    def get_detected_language(self):
        """
        Get the detected language.
        
        Returns:
            str: Detected language code
        """
        return self.detected_language
    
    def get_detected_emotion(self):
        """
        Get the detected emotion.
        
        Returns:
            str: Detected emotion
        """
        return self.detected_emotion


class SimulatedSenseVoiceModel:
    """
    A simulated SenseVoice model for demonstration purposes.
    In a real implementation, this would be replaced with the actual SenseVoice model.
    """
    
    def __init__(self):
        """Initialize the simulated model."""
        self.processor = None
        self.sample_responses = [
            {"text": "Hello, how are you?", "language": "en", "emotion": "neutral"},
            {"text": "I'm really excited about this project!", "language": "en", "emotion": "happy"},
            {"text": "I'm not sure if this is working correctly.", "language": "en", "emotion": "concerned"},
            {"text": "Can you help me with this problem?", "language": "en", "emotion": "neutral"},
            {"text": "This is absolutely unacceptable!", "language": "en", "emotion": "angry"},
            {"text": "I'm feeling a bit down today.", "language": "en", "emotion": "sad"}
        ]
    
    def transcribe(self, audio, is_final=False, cache=None):
        """
        Simulate transcription by returning a predefined response.
        
        Args:
            audio (numpy.ndarray): Audio data
            is_final (bool): Whether this is the final processing
            cache (any): Cached features for streaming
            
        Returns:
            dict: Simulated transcription result
        """
        # For simulation, we'll return a random sample response
        # In a real implementation, this would process the audio and return actual transcription
        import random
        
        # Simulate incremental recognition by returning partial text
        if not is_final:
            sample = random.choice(self.sample_responses)
            words = sample["text"].split()
            # Return a random number of words to simulate partial recognition
            partial_length = random.randint(1, len(words))
            partial_text = " ".join(words[:partial_length])
            return {
                "text": partial_text,
                "language": sample["language"],
                "emotion": sample["emotion"]
            }
        else:
            # Return a complete sample for final recognition
            return random.choice(self.sample_responses)
