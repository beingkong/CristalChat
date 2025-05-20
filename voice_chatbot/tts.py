"""
Text-to-Speech (TTS) module using F5-TTS for voice synthesis.
"""

import os
import numpy as np
import soundfile as sf
import tempfile
from typing import Dict, List, Optional, Tuple, Union

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False


class TTSModule:
    """
    Text-to-Speech module that uses F5-TTS to convert text to speech.
    """
    
    def __init__(self, model_path=None, device="cuda", voice_ref_path=None):
        """
        Initialize the TTS module with F5-TTS model.
        
        Args:
            model_path (str, optional): Path to F5-TTS model. If None, will use default.
            device (str): Device to run the model on ("cuda" or "cpu")
            voice_ref_path (str, optional): Path to reference voice audio file for voice cloning
        """
        # Check if CUDA is available and fall back to CPU if not
        if device == "cuda" and (not TORCH_AVAILABLE or (TORCH_AVAILABLE and not torch.cuda.is_available())):
            print("TTS: CUDA not available, falling back to CPU")
            device = "cpu"
            
        self.device = device
        self.model = None
        self.voice_ref_path = voice_ref_path
        self.default_emotion = "neutral"
        self.default_speed = 1.0
        
        # Load model (in a real implementation, this would use the actual F5-TTS model)
        self._load_model(model_path)
        print(f"TTS: Initialized with F5-TTS model on {device}")
    
    def _load_model(self, model_path):
        """
        Load the F5-TTS model.
        
        In a real implementation, this would use:
        import f5tts
        self.model = f5tts.load_model(model_path or "f5/f5-tts-base", device=self.device)
        
        For this implementation, we'll simulate the model loading.
        """
        # Simulate model loading
        print("TTS: Loading F5-TTS model (simulated)")
        # In a real implementation, this would load the actual model
        
        # For simulation purposes
        self.model = SimulatedF5TTSModel()
    
    def synthesize(self, text, voice_ref=None, emotion=None, speed=None):
        """
        Synthesize speech from text.
        
        Args:
            text (str): Text to synthesize
            voice_ref (str, optional): Path to reference voice audio file
            emotion (str, optional): Emotion to apply to the speech
            speed (float, optional): Speech speed factor
            
        Returns:
            numpy.ndarray: Audio data
            int: Sample rate
        """
        # Use defaults if parameters not provided
        voice_ref = voice_ref or self.voice_ref_path
        emotion = emotion or self.default_emotion
        speed = speed or self.default_speed
        
        print(f"TTS: Synthesizing text: '{text[:30]}...' with emotion: {emotion}, speed: {speed}")
        
        # In a real implementation, this would call the F5-TTS model
        # audio = self.model.synthesize(
        #     text=text,
        #     voice_ref=voice_ref,
        #     emotion=emotion,
        #     speed=speed
        # )
        
        # For simulation, we'll use our simulated model
        audio, sample_rate = self.model.synthesize(text, voice_ref, emotion, speed)
        
        return audio, sample_rate
    
    def synthesize_to_file(self, text, output_path, voice_ref=None, emotion=None, speed=None):
        """
        Synthesize speech from text and save to file.
        
        Args:
            text (str): Text to synthesize
            output_path (str): Path to save the audio file
            voice_ref (str, optional): Path to reference voice audio file
            emotion (str, optional): Emotion to apply to the speech
            speed (float, optional): Speech speed factor
            
        Returns:
            str: Path to the saved audio file
        """
        audio, sample_rate = self.synthesize(text, voice_ref, emotion, speed)
        
        # Save audio to file
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    def synthesize_ssml(self, ssml_text, voice_ref=None):
        """
        Synthesize speech from SSML text.
        This allows more control over the speech synthesis.
        
        Args:
            ssml_text (str): SSML text to synthesize
            voice_ref (str, optional): Path to reference voice audio file
            
        Returns:
            numpy.ndarray: Audio data
            int: Sample rate
        """
        print(f"TTS: Synthesizing SSML: '{ssml_text[:30]}...'")
        
        # In a real implementation, this would parse SSML and call the F5-TTS model
        # with appropriate parameters
        
        # For simulation, we'll use our simulated model
        audio, sample_rate = self.model.synthesize_ssml(ssml_text, voice_ref)
        
        return audio, sample_rate
    
    def set_voice(self, voice_ref_path):
        """
        Set the reference voice for future synthesis.
        
        Args:
            voice_ref_path (str): Path to reference voice audio file
        """
        self.voice_ref_path = voice_ref_path
        print(f"TTS: Set voice reference to {voice_ref_path}")
    
    def set_default_emotion(self, emotion):
        """
        Set the default emotion for future synthesis.
        
        Args:
            emotion (str): Default emotion
        """
        self.default_emotion = emotion
        print(f"TTS: Set default emotion to {emotion}")
    
    def set_default_speed(self, speed):
        """
        Set the default speed for future synthesis.
        
        Args:
            speed (float): Default speed factor
        """
        self.default_speed = speed
        print(f"TTS: Set default speed to {speed}")


class SimulatedF5TTSModel:
    """
    A simulated F5-TTS model for demonstration purposes.
    In a real implementation, this would be replaced with the actual F5-TTS model.
    """
    
    def __init__(self):
        """Initialize the simulated model."""
        self.sample_rate = 24000
        
        # Create a temporary directory for storing audio files
        self.temp_dir = tempfile.mkdtemp()
        print(f"TTS: Created temporary directory for audio files: {self.temp_dir}")
    
    def synthesize(self, text, voice_ref=None, emotion=None, speed=None):
        """
        Simulate speech synthesis by generating a sine wave.
        
        Args:
            text (str): Text to synthesize
            voice_ref (str, optional): Path to reference voice audio file
            emotion (str, optional): Emotion to apply to the speech
            speed (float, optional): Speech speed factor
            
        Returns:
            numpy.ndarray: Audio data
            int: Sample rate
        """
        # For simulation, we'll generate a simple sine wave
        # The length of the audio depends on the text length
        duration = len(text) * 0.05  # 50ms per character
        
        # Adjust duration based on speed
        if speed:
            duration /= speed
        
        # Generate a sine wave
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Base frequency
        freq = 220.0  # A3
        
        # Adjust frequency based on emotion
        if emotion == "happy":
            freq *= 1.2
        elif emotion == "sad":
            freq *= 0.8
        elif emotion == "angry":
            freq *= 1.5
        
        # Generate the sine wave
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add some variation to simulate speech
        for i, char in enumerate(text):
            if i < len(t) // len(text):
                idx = i * (len(t) // len(text))
                end_idx = (i + 1) * (len(t) // len(text))
                if char in "aeiou":
                    audio[idx:end_idx] *= 1.2
                elif char in "ptkbdg":
                    audio[idx:end_idx] *= 0.8
        
        return audio, self.sample_rate
    
    def synthesize_ssml(self, ssml_text, voice_ref=None):
        """
        Simulate SSML speech synthesis.
        
        Args:
            ssml_text (str): SSML text to synthesize
            voice_ref (str, optional): Path to reference voice audio file
            
        Returns:
            numpy.ndarray: Audio data
            int: Sample rate
        """
        # For simulation, we'll just strip SSML tags and synthesize as plain text
        import re
        plain_text = re.sub(r'<[^>]+>', '', ssml_text)
        return self.synthesize(plain_text, voice_ref)
    
    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"TTS: Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"TTS: Error removing temporary directory: {e}")
