import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
from google.cloud import speech_v1
from google.cloud import texttospeech

class AudioRecorder:
    def __init__(self):
        self.sample_rate = 44100
        self.channels = 1
        self.recording = []
        self.is_recording = False

    def start_recording(self):
        """Start recording audio"""
        self.recording = []
        self.is_recording = True
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                self.recording.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        if not self.recording:
            return None
            
        # Combine all recorded chunks
        audio_data = np.concatenate(self.recording, axis=0)
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes per sample
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            return temp_file.name

def transcribe_audio(audio_file, language_code="en-US", speech_client=None):
    """Convert speech to text using Google Speech-to-Text"""
    if not speech_client:
        return "Speech-to-Text service not available. Please check your credentials."
    
    try:
        with open(audio_file, 'rb') as audio:
            content = audio.read()
        
        audio = speech_v1.RecognitionAudio(content=content)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code=language_code,
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            return "No speech detected"
            
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
            
        return transcript.strip()
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(audio_file)
        except:
            pass

def text_to_speech(text, language_code="en-US", voice_name="en-US-Studio-O", tts_client=None):
    """Convert text to speech using Google Text-to-Speech"""
    if not tts_client:
        return None
    
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.audio_content)
            return temp_file.name
            
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None 