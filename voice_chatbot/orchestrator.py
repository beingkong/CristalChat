"""
Orchestrator module for coordinating the voice chatbot components.
"""

import asyncio
import queue
import threading
import os
import time
from typing import Dict, List, Optional, Union

from voice_chatbot.vad import VADModule
from voice_chatbot.asr import ASRModule
from voice_chatbot.ser import SERModule
from voice_chatbot.llm import QwenLLMModule
from voice_chatbot.tts import TTSModule
from voice_chatbot.audio_player import AudioPlayerThread
from voice_chatbot.utils.text_utils import clean_text, compose_prompt, save_to_history


class OrchestratorModule:
    """
    Orchestrator module that coordinates all components of the voice chatbot.
    """
    
    def __init__(self, prompt_template_path=None, voice_ref_path=None, use_local_mode=False):
        """
        Initialize the orchestrator.
        
        Args:
            prompt_template_path (str, optional): Path to prompt template file
            voice_ref_path (str, optional): Path to reference voice audio file
            use_local_mode (bool): Whether to use local mode (text input/output)
        """
        self.vad = None
        self.asr = None
        self.ser = None
        self.llm = None
        self.tts = None
        self.audio_player = None
        
        self.prompt_template_path = prompt_template_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "prompts", "assistant_prompt.txt"
        )
        self.voice_ref_path = voice_ref_path
        self.use_local_mode = use_local_mode
        
        # TTS queue for streaming synthesis
        self.tts_queue = asyncio.Queue()
        
        # Conversation history
        self.context_history = []
        
        # User interrupt flag
        self.user_interrupt = False
        
        # Initialize components
        self._initialize_components()
        
        print("Orchestrator: Initialized all components")
    
    def _initialize_components(self):
        """Initialize all components of the voice chatbot."""
        # Initialize LLM
        self.llm = QwenLLMModule(prompt_template_path=self.prompt_template_path)
        print("Orchestrator: Initialized LLM module")
        
        # Initialize TTS
        self.tts = TTSModule(voice_ref_path=self.voice_ref_path)
        print("Orchestrator: Initialized TTS module")
        
        if not self.use_local_mode:
            # Initialize VAD
            self.vad = VADModule()
            print("Orchestrator: Initialized VAD module")
            
            # Initialize ASR
            self.asr = ASRModule()
            print("Orchestrator: Initialized ASR module")
            
            # Initialize SER
            self.ser = SERModule()
            print("Orchestrator: Initialized SER module")
            
            # Initialize Audio Player
            self.audio_player = AudioPlayerThread()
            self.audio_player.start()
            print("Orchestrator: Started Audio Player thread")
        
        # Initialize Audio Player
        self.audio_player = AudioPlayerThread()
        self.audio_player.start()
        print("Orchestrator: Started Audio Player thread")
    
    async def start_audio_player_thread(self, tts_queue, audio_player):
        """
        Start the audio player thread that consumes from the TTS queue.
        
        Args:
            tts_queue (asyncio.Queue): Queue of text to be synthesized
            audio_player (AudioPlayerThread): Audio player instance
        """
        print("Orchestrator: Starting audio player consumer task")
        
        while True:
            try:
                # Get text from queue
                text = await tts_queue.get()
                
                if text is None:
                    # None is a signal to stop
                    tts_queue.task_done()
                    break
                
                # Synthesize speech
                emotion = self.ser.get_detected_emotion() if self.ser else "neutral"
                audio, sample_rate = self.tts.synthesize(text, emotion=emotion)
                
                # Play audio
                self.audio_player.play(audio, sample_rate)
                
                # Mark task as done
                tts_queue.task_done()
            
            except asyncio.CancelledError:
                # Task was cancelled
                break
            
            except Exception as e:
                print(f"Orchestrator: Error in audio player consumer: {e}")
        
        print("Orchestrator: Audio player consumer task stopped")
    
    async def main_loop(self):
        """
        Main event loop for the voice chatbot.
        This implements the core conversation flow.
        """
        print("Orchestrator: Starting main loop")
        
        # Start audio player consumer task
        audio_player_task = asyncio.create_task(
            self.start_audio_player_thread(self.tts_queue, self.audio_player)
        )
        
        try:
            while True:
                # Reset user interrupt flag
                self.user_interrupt = False
                
                print("Orchestrator: Waiting for user to speak...")
                
                # 1. Wait for user to speak
                await self.vad.wait_for_speech_start()
                
                print("Orchestrator: User started speaking")
                
                # 2. Start ASR and SER
                self.asr.start_recognition()
                self.ser.start_detection()
                
                # 3. Process audio frames while user is speaking
                while self.vad.is_speaking():
                    # Get audio frame
                    audio_frame = self.vad.get_audio_frame()
                    
                    if audio_frame is not None:
                        # Process with ASR and SER
                        partial_text = self.asr.consume_audio(audio_frame)
                        self.ser.consume_audio(audio_frame)
                        
                        # Optionally feed partial text to LLM
                        if partial_text:
                            self.llm.feed_partial_input(partial_text)
                    
                    # Yield control to allow other tasks to run
                    await asyncio.sleep(0)
                
                print("Orchestrator: User stopped speaking")
                
                # 4. Get final ASR and SER results
                user_text = self.asr.get_result_text()
                emotion = self.ser.get_detected_emotion()
                language = self.asr.get_detected_language()
                
                print(f"Orchestrator: ASR result: '{user_text}'")
                print(f"Orchestrator: Detected emotion: {emotion}")
                print(f"Orchestrator: Detected language: {language}")
                
                # Save to conversation history
                self.context_history.append(save_to_history("user", user_text))
                
                # 5. Compose prompt for LLM
                prompt = compose_prompt(user_text, emotion, self.context_history)
                
                # 6. Generate response with LLM
                current_sentence = ""
                
                print("Orchestrator: Generating response with LLM")
                
                async for chunk in self.llm.generate(prompt):
                    text_chunk = chunk.get_text()
                    
                    if text_chunk:
                        current_sentence += text_chunk
                        
                        # Check for complete sentences
                        sentences = current_sentence.split('\n')
                        
                        for s in sentences[:-1]:
                            reply_sentence = s.strip()
                            
                            if reply_sentence:
                                # Clean text for TTS
                                cleaned = clean_text(reply_sentence)
                                
                                # Add to TTS queue
                                await self.tts_queue.put(cleaned)
                                
                                # Save to history
                                self.context_history.append(save_to_history("assistant", reply_sentence))
                        
                        # Keep the incomplete sentence
                        current_sentence = sentences[-1]
                    
                    # Check for user interrupt
                    if self.vad.detect_voice_during_output():
                        self.user_interrupt = True
                        break
                    
                    # Yield control
                    await asyncio.sleep(0)
                
                # 7. Handle user interrupt
                if self.user_interrupt:
                    print("Orchestrator: User interrupted, stopping generation")
                    
                    # Stop LLM generation
                    self.llm.stop_generation()
                    
                    # Clear TTS queue
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    
                    # Stop audio playback
                    self.audio_player.clear_pending()
                    self.audio_player.player.stop_playback()
                    
                    # Continue to next iteration (wait for new user input)
                    continue
                
                # 8. Process any remaining text
                if current_sentence.strip():
                    cleaned = clean_text(current_sentence.strip())
                    await self.tts_queue.put(cleaned)
                    self.context_history.append(save_to_history("assistant", current_sentence.strip()))
                
                print("Orchestrator: Response generation complete")
                
                # Wait a bit before listening for the next input
                # This gives time for TTS to finish
                await asyncio.sleep(0.5)
        
        except asyncio.CancelledError:
            print("Orchestrator: Main loop cancelled")
        
        except Exception as e:
            print(f"Orchestrator: Error in main loop: {e}")
        
        finally:
            # Clean up
            print("Orchestrator: Cleaning up")
            
            # Cancel audio player task
            audio_player_task.cancel()
            try:
                await audio_player_task
            except asyncio.CancelledError:
                pass
            
            # Stop audio player
            self.audio_player.stop()
    
    async def process_text_input(self, text_input):
        """
        Process text input and generate response.
        
        Args:
            text_input (str): User's text input
        
        Returns:
            str: Bot's response
        """
        # Save to conversation history
        self.context_history.append(save_to_history("user", text_input))
        
        # Compose prompt for LLM
        prompt = compose_prompt(text_input, "neutral", self.context_history)
        
        # Generate response with LLM
        current_sentence = ""
        response = ""
        
        async for chunk in self.llm.generate(prompt):
            text_chunk = chunk.get_text()
            
            if text_chunk:
                current_sentence += text_chunk
                
                # Check for complete sentences
                sentences = current_sentence.split('\n')
                
                for s in sentences[:-1]:
                    reply_sentence = s.strip()
                    
                    if reply_sentence:
                        response += reply_sentence + "\n"
                        self.context_history.append(save_to_history("assistant", reply_sentence))
                
                # Keep the incomplete sentence
                current_sentence = sentences[-1]
        
        # Process any remaining text
        if current_sentence.strip():
            response += current_sentence.strip()
            self.context_history.append(save_to_history("assistant", current_sentence.strip()))
        
        return response

    def start(self):
        """Start the voice chatbot."""
        print("Orchestrator: Starting voice chatbot")
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run main loop
            loop.run_until_complete(self.main_loop())
        
        except KeyboardInterrupt:
            print("Orchestrator: Keyboard interrupt received")
        
        finally:
            # Clean up
            loop.close()
            print("Orchestrator: Voice chatbot stopped")
    
    def stop(self):
        """Stop the voice chatbot."""
        print("Orchestrator: Stopping voice chatbot")
        
        # Stop audio player
        if self.audio_player:
            self.audio_player.stop()
        
        # Additional cleanup if needed
        
        print("Orchestrator: Voice chatbot stopped")


def start_audio_player_thread(tts_queue, audio_player):
    """
    Start the audio player thread that consumes from the TTS queue.
    This is a synchronous version for use in a separate thread.
    
    Args:
        tts_queue (queue.Queue): Queue of text to be synthesized
        audio_player (AudioPlayerThread): Audio player instance
    """
    print("Starting audio player thread")
    
    while True:
        try:
            # Get text from queue with timeout
            text = tts_queue.get(timeout=0.5)
            
            if text is None:
                # None is a signal to stop
                tts_queue.task_done()
                break
            
            # Get TTS module (in a real implementation, this would be passed in)
            tts = TTSModule()
            
            # Synthesize speech
            audio, sample_rate = tts.synthesize(text)
            
            # Play audio
            audio_player.play(audio, sample_rate)
            
            # Mark task as done
            tts_queue.task_done()
        
        except queue.Empty:
            # Queue is empty, continue waiting
            pass
        
        except Exception as e:
            print(f"Error in audio player thread: {e}")
    
    print("Audio player thread stopped")
