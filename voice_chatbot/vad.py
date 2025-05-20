"""
Voice Activity Detection (VAD) module for detecting speech in audio streams.
"""

import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False


class VADModule:
    """
    Voice Activity Detection module that monitors microphone input
    and detects when speech starts and ends.
    """
    
    def __init__(self, sample_rate=16000, frame_duration_ms=30, 
                 speech_energy_threshold=0.01, speech_start_frames=3,
                 speech_end_frames=15, max_speech_duration_sec=30):
        """
        Initialize the VAD module.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            frame_duration_ms (int): Duration of each audio frame in milliseconds
            speech_energy_threshold (float): Energy threshold for speech detection
            speech_start_frames (int): Number of consecutive frames above threshold to trigger speech start
            speech_end_frames (int): Number of consecutive frames below threshold to trigger speech end
            max_speech_duration_sec (int): Maximum duration of speech segment in seconds
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.speech_energy_threshold = speech_energy_threshold
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.max_speech_duration_sec = max_speech_duration_sec
        self.max_speech_frames = int(max_speech_duration_sec * 1000 / frame_duration_ms)
        
        # State variables
        self.is_speaking_flag = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.current_speech_frames = 0
        self.speech_detected_event = asyncio.Event()
        
        # Audio buffer for storing captured frames
        self.audio_buffer = queue.Queue()
        
        # Thread for continuous audio capture
        self.audio_thread = None
        self.stop_audio_thread = threading.Event()
        
        # Start audio capture thread
        self.start_audio_capture()
    
    def start_audio_capture(self):
        """Start the audio capture thread."""
        self.stop_audio_thread.clear()
        self.audio_thread = threading.Thread(target=self._audio_capture_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_audio_capture(self):
        """Stop the audio capture thread."""
        if self.audio_thread and self.audio_thread.is_alive():
            self.stop_audio_thread.set()
            self.audio_thread.join(timeout=1.0)
    
    def _audio_capture_loop(self):
        """Continuously capture audio from microphone and process for VAD."""
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream to process incoming audio frames."""
            if status:
                print(f"Audio callback status: {status}")
            
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_frame = indata[:, 0].copy()
            else:
                audio_frame = indata[:, 0].copy()
            
            # Calculate energy
            energy = np.mean(audio_frame ** 2)
            
            # Speech detection logic
            if energy > self.speech_energy_threshold:
                self.consecutive_speech_frames += 1
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_frames += 1
                self.consecutive_speech_frames = 0
            
            # State transitions
            if not self.is_speaking_flag and self.consecutive_speech_frames >= self.speech_start_frames:
                # Speech start detected
                self.is_speaking_flag = True
                self.current_speech_frames = 0
                self.speech_detected_event.set()
                print("Speech detected - started recording")
            
            if self.is_speaking_flag:
                # Add frame to buffer while speaking
                self.audio_buffer.put(audio_frame)
                self.current_speech_frames += 1
                
                # Check for speech end conditions
                if (self.consecutive_silence_frames >= self.speech_end_frames or 
                        self.current_speech_frames >= self.max_speech_frames):
                    # Speech end detected
                    self.is_speaking_flag = False
                    print("Speech detected - stopped recording")
        
        try:
            # List available audio devices for debugging
            try:
                devices = sd.query_devices()
                print("Available audio devices:")
                for i, device in enumerate(devices):
                    print(f"  {i}: {device['name']}")
            except Exception as e:
                print(f"Could not query audio devices: {e}")
            
            # Try to use default input device
            try:
                with sd.InputStream(callback=audio_callback,
                                   channels=1,
                                   samplerate=self.sample_rate,
                                   blocksize=self.frame_size):
                    print("VAD: Listening for speech...")
                    while not self.stop_audio_thread.is_set():
                        self.stop_audio_thread.wait(timeout=0.1)
            except Exception as e:
                print(f"Error with default audio device: {e}")
                
                # If no audio device is available, simulate audio input for testing
                print("No audio device available. Using simulated audio input.")
                self._simulate_audio_input()
                
        except Exception as e:
            print(f"Error in audio capture: {e}")
            
    def _simulate_audio_input(self):
        """Simulate audio input for testing when no microphone is available."""
        import time
        import random
        
        print("VAD: Using simulated audio input")
        
        while not self.stop_audio_thread.is_set():
            # Simulate occasional speech detection
            if random.random() < 0.01:  # 1% chance per iteration to "detect" speech
                print("Simulated speech detected - started recording")
                self.is_speaking_flag = True
                self.current_speech_frames = 0
                self.speech_detected_event.set()
                
                # Generate some random audio frames
                for _ in range(50):  # Simulate about 1.5 seconds of speech
                    if self.stop_audio_thread.is_set():
                        break
                    
                    # Create a random audio frame (white noise)
                    audio_frame = np.random.normal(0, 0.1, self.frame_size)
                    self.audio_buffer.put(audio_frame)
                    self.current_speech_frames += 1
                    time.sleep(0.03)  # Simulate frame duration
                
                # End the simulated speech
                self.is_speaking_flag = False
                print("Simulated speech detected - stopped recording")
            
            time.sleep(0.1)  # Check for new simulated speech every 100ms
    
    async def wait_for_speech_start(self):
        """
        Wait asynchronously until speech is detected.
        
        Returns:
            bool: True when speech is detected
        """
        # Clear previous event state
        self.speech_detected_event.clear()
        
        # Wait for speech detection event
        await self.speech_detected_event.wait()
        return True
    
    def is_speaking(self):
        """
        Check if user is currently speaking.
        
        Returns:
            bool: True if user is speaking, False otherwise
        """
        return self.is_speaking_flag
    
    def get_audio_frame(self):
        """
        Get the next audio frame from the buffer.
        
        Returns:
            numpy.ndarray: Audio frame data or None if no data available
        """
        try:
            return self.audio_buffer.get(block=False)
        except queue.Empty:
            return None
    
    def detect_voice_during_output(self):
        """
        Check if user voice is detected during system output (for barge-in detection).
        This is a simplified version that just checks the speaking flag.
        In a real implementation, this would need echo cancellation to avoid
        detecting the system's own output as user speech.
        
        Returns:
            bool: True if user voice detected during output
        """
        # In a real implementation, this would use a separate VAD instance
        # with echo cancellation to avoid detecting system output
        return self.is_speaking_flag
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.stop_audio_capture()
