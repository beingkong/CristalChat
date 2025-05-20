"""
Audio Player module for playing synthesized speech.
"""

import queue
import threading
import time
import numpy as np
import sounddevice as sd


class AudioOutputPlayer:
    """
    Audio Output Player for playing synthesized speech.
    Runs in a separate thread to allow concurrent audio playback.
    """
    
    def __init__(self, sample_rate=24000, channels=1, dtype='float32'):
        """
        Initialize the audio player.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
            dtype (str): Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        # Stream for audio playback
        self.stream = None
        
        # Flag to indicate if playback should stop
        self.stop_flag = threading.Event()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Currently playing audio data
        self.current_audio = None
        
        print(f"AudioPlayer: Initialized with sample rate {sample_rate}Hz")
    
    def play(self, audio_data, sample_rate=None, blocking=False):
        """
        Play audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio data to play
            sample_rate (int, optional): Sample rate of the audio data
            blocking (bool): Whether to block until playback is complete
            
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if audio_data is None or len(audio_data) == 0:
            print("AudioPlayer: No audio data to play")
            return False
        
        # Use provided sample rate or default
        sr = sample_rate or self.sample_rate
        
        # Stop any current playback
        self.stop_playback()
        
        # Reset stop flag
        self.stop_flag.clear()
        
        with self.lock:
            self.current_audio = audio_data
        
        # Start playback
        try:
            # Create a new stream for each playback
            self.stream = sd.OutputStream(
                samplerate=sr,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"AudioPlayer: Started playback of {len(audio_data)} samples")
            
            if blocking:
                # Wait until playback is complete
                while not self.stop_flag.is_set() and self.stream.active:
                    time.sleep(0.1)
                
                # Close the stream
                self.stream.close()
                self.stream = None
            
            return True
        
        except Exception as e:
            print(f"AudioPlayer: Error starting playback: {e}")
            return False
    
    def _audio_callback(self, outdata, frames, time, status):
        """
        Callback for audio output stream.
        
        Args:
            outdata (numpy.ndarray): Output buffer to fill with audio data
            frames (int): Number of frames to fill
            time (CData): Timestamps for the audio chunk
            status (CallbackFlags): Status flags
        """
        if status:
            print(f"AudioPlayer: Status: {status}")
        
        with self.lock:
            if self.current_audio is None or self.stop_flag.is_set():
                # No audio data or stop requested, fill with zeros
                outdata.fill(0)
                raise sd.CallbackStop
            
            # Copy audio data to output buffer
            if len(self.current_audio) > frames:
                # More data than needed for this buffer
                outdata[:] = self.current_audio[:frames].reshape(-1, 1)
                self.current_audio = self.current_audio[frames:]
            else:
                # Less data than needed, fill with zeros and stop
                outdata[:len(self.current_audio)] = self.current_audio.reshape(-1, 1)
                outdata[len(self.current_audio):] = 0
                self.current_audio = None
                self.stop_flag.set()
                raise sd.CallbackStop
    
    def stop_playback(self):
        """Stop any ongoing audio playback."""
        self.stop_flag.set()
        
        with self.lock:
            self.current_audio = None
        
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
                self.stream = None
                print("AudioPlayer: Stopped playback")
            except Exception as e:
                print(f"AudioPlayer: Error stopping playback: {e}")
    
    def is_playing(self):
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing, False otherwise
        """
        return self.stream is not None and self.stream.active and not self.stop_flag.is_set()


class AudioPlayerThread:
    """
    Thread for playing audio from a queue.
    This allows for streaming TTS output as it's generated.
    """
    
    def __init__(self, sample_rate=24000, channels=1, dtype='float32'):
        """
        Initialize the audio player thread.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
            dtype (str): Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        # Queue for audio data
        self.audio_queue = queue.Queue()
        
        # Audio player
        self.player = AudioOutputPlayer(sample_rate, channels, dtype)
        
        # Thread for audio playback
        self.thread = None
        self.stop_thread = threading.Event()
        
        print("AudioPlayerThread: Initialized")
    
    def start(self):
        """Start the audio player thread."""
        if self.thread is not None and self.thread.is_alive():
            print("AudioPlayerThread: Thread already running")
            return
        
        self.stop_thread.clear()
        self.thread = threading.Thread(target=self._player_thread)
        self.thread.daemon = True
        self.thread.start()
        print("AudioPlayerThread: Started thread")
    
    def stop(self):
        """Stop the audio player thread."""
        if self.thread is not None and self.thread.is_alive():
            self.stop_thread.set()
            self.player.stop_playback()
            self.clear_pending()
            self.thread.join(timeout=1.0)
            self.thread = None
            print("AudioPlayerThread: Stopped thread")
    
    def _player_thread(self):
        """Thread function for audio playback."""
        print("AudioPlayerThread: Thread running")
        
        while not self.stop_thread.is_set():
            try:
                # Get audio data from queue with timeout
                audio_data, sample_rate = self.audio_queue.get(timeout=0.5)
                
                # Play audio
                if audio_data is not None and len(audio_data) > 0:
                    self.player.play(audio_data, sample_rate, blocking=True)
                
                # Mark task as done
                self.audio_queue.task_done()
            
            except queue.Empty:
                # Queue is empty, continue waiting
                pass
            
            except Exception as e:
                print(f"AudioPlayerThread: Error in player thread: {e}")
        
        print("AudioPlayerThread: Thread exiting")
    
    def play(self, audio_data, sample_rate=None):
        """
        Add audio data to the playback queue.
        
        Args:
            audio_data (numpy.ndarray): Audio data to play
            sample_rate (int, optional): Sample rate of the audio data
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        # Use provided sample rate or default
        sr = sample_rate or self.sample_rate
        
        # Add to queue
        self.audio_queue.put((audio_data, sr))
        print(f"AudioPlayerThread: Added {len(audio_data)} samples to queue")
    
    def clear_pending(self):
        """Clear all pending audio data from the queue."""
        try:
            while True:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except queue.Empty:
            pass
        
        print("AudioPlayerThread: Cleared pending audio data")
    
    def is_playing(self):
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing or queued, False otherwise
        """
        return self.player.is_playing() or not self.audio_queue.empty()
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.stop()
