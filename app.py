# Must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Genie AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
from google.cloud import speech_v1
from google.cloud import texttospeech
import tempfile
import wave
import io
import sounddevice as sd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from screen_analyzer import ScreenAnalyzer

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE")

speech_client = None
tts_client = None
model = None

# Initialize clients silently
try:
    if GOOGLE_CREDENTIALS_FILE and os.path.exists(GOOGLE_CREDENTIALS_FILE):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_FILE
        speech_client = speech_v1.SpeechClient()
        tts_client = texttospeech.TextToSpeechClient()
except Exception:
    pass

# Gemini - Silent initialization
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    pass

LANGUAGES = {
    "English": {"code": "en-US", "voice": "en-US-Studio-O"},
    "Hindi": {"code": "hi-IN", "voice": "hi-IN-Standard-A"},
    "Spanish": {"code": "es-ES", "voice": "es-ES-Standard-A"},
    "French": {"code": "fr-FR", "voice": "fr-FR-Standard-A"},
    "German": {"code": "de-DE", "voice": "de-DE-Standard-A"},
    "Japanese": {"code": "ja-JP", "voice": "ja-JP-Standard-A"},
    "Chinese": {"code": "zh-CN", "voice": "zh-CN-Standard-A"}
}

class AudioRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_data = []
        self.stream = None
        
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.audio_data = []
        
        # Start the input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback
        )
        self.stream.start()
        return True
    
    def stop_recording(self):
        """Stop recording and return the audio data"""
        if self.stream:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            return self.get_wav_data()
        return None
    
    def callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def get_wav_data(self):
        """Convert recorded audio to WAV format"""
        if not self.audio_data:
            return None
        
        # Combine all audio chunks
        audio = np.concatenate(self.audio_data)
        
        # Convert to WAV format
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
            return wav_io.getvalue()

def transcribe_audio(audio_data, language_code="en-US"):
    """Transcribe audio using Google Cloud Speech-to-Text"""
    if not speech_client:
        return "Speech-to-Text client not available"
    
    try:
        # Create recognition audio
        audio = speech_v1.RecognitionAudio(content=audio_data)
        
        # Configure recognition
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="default",
        )
        
        # Detect speech
        response = speech_client.recognize(
            config=config,
            audio=audio
        )
        
        # Get transcription
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            return transcript
        else:
            return "No speech detected"
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def get_ai_response(text, language="English"):
    """Get response from Gemini"""
    if not model:
        return "AI model not available. Please check your GOOGLE_API_KEY."
    
    try:
        # Simplified prompt
        if language == "English":
            prompt = text
        else:
            prompt = f"Please respond in {language} to this question: {text}"
        
        # Generate response with specific configuration
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=150,
            temperature=0.7,
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Check if response was blocked or empty
        if not response.text:
            if hasattr(response, 'prompt_feedback'):
                return f"Response blocked due to safety filters: {response.prompt_feedback}"
            else:
                return "Empty response received from AI model."
        
        return response.text.strip()
        
    except Exception as e:
        # Return the actual error for debugging
        error_msg = str(e)
        return f"AI Error: {error_msg}"

def text_to_speech(text, language_code="en-US", voice_name="en-US-Studio-O"):
    """Convert text to speech using Google Cloud TTS"""
    if not tts_client:
        return None
    
    try:
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Configure audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # Generate speech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.audio_content)
            return temp_audio.name
            
    except Exception as e:
        return None

def initialize_session_state():
    if "recorder" not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = "freedraw"
    if "selected_color" not in st.session_state:
        st.session_state.selected_color = "#000000"
    if "stroke_width" not in st.session_state:
        st.session_state.stroke_width = 3
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = ScreenAnalyzer()

def main():
    initialize_session_state()
    
    # Sidebar for voice assistant
    with st.sidebar:
        # Language selection at the top
        selected_language = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )
        
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Google Assistant-like button in the center
        button_style = """
            <style>
            div.stButton > button {
                border-radius: 50%;
                height: 80px;
                width: 80px;
                font-size: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
                background-color: #4285F4;
                color: white;
                border: none;
            }
            div.stButton > button:hover {
                background-color: #3367D6;
                border: none;
            }
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            button_label = "🎤" if not st.session_state.is_recording else "⏹️"
            if st.button(button_label, key="toggle_recording"):
                if not st.session_state.is_recording:
                    # Start recording
                    st.session_state.is_recording = True
                    st.session_state.recorder.start_recording()
                    st.rerun()
                else:
                    # Stop recording and process silently
                    st.session_state.is_recording = False
                    audio_data = st.session_state.recorder.stop_recording()
                    
                    if audio_data:
                        # Get language settings
                        lang_code = LANGUAGES[st.session_state.selected_language]["code"]
                        voice_name = LANGUAGES[st.session_state.selected_language]["voice"]
                        
                        # Process voice silently in background
                        # Convert speech to text
                        text = transcribe_audio(audio_data, lang_code)
                        
                        if text and text != "No speech detected":
                            # Get AI response
                            response = get_ai_response(text, st.session_state.selected_language)
                            st.session_state.last_response = response
                            
                            # Generate speech
                            audio_file = text_to_speech(response, lang_code, voice_name)
                            
                            if audio_file:
                                st.session_state.audio_file = audio_file
                    st.rerun()
        
        # Status indicator in the center
        status_text = "🔴 Listening..." if st.session_state.is_recording else "Tap to speak"
        st.markdown(f"<div style='text-align: center; margin-top: 10px;'>{status_text}</div>", unsafe_allow_html=True)
        
        # Add spacing before response
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Auto-play audio response and show text (Google Assistant style)
        if st.session_state.last_response and st.session_state.audio_file:
            # Show response text
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0;'>{st.session_state.last_response}</div>", unsafe_allow_html=True)
            # Auto-play audio
            st.audio(st.session_state.audio_file, format="audio/mp3", autoplay=True)
    
    # Main area for whiteboard
    st.markdown("<h1 style='text-align: left; color: #1E88E5;'>Genie AI</h1>", unsafe_allow_html=True)
    
    # Toolbar with drawing options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        mode = st.radio("", ["Draw", "Erase", "Clear"], horizontal=True)
        st.session_state.selected_color = st.color_picker("Select Color", st.session_state.selected_color)
        st.session_state.stroke_width = st.slider("Stroke Width", 1, 20, st.session_state.stroke_width)
    
    # Drawing mode handling
    drawing_mode = "freedraw"
    if mode == "Erase":
        drawing_mode = "freedraw"
        st.session_state.selected_color = "#FFFFFF"
    elif mode == "Clear":
        st.session_state.selected_color = "#FFFFFF"
        st.session_state.stroke_width = 50
    
    # Drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=st.session_state.stroke_width,
        stroke_color=st.session_state.selected_color,
        background_color="#FFFFFF",
        width=800,
        height=500,
        drawing_mode=drawing_mode,
        key="whiteboard_canvas"
    )
    
    # Analyze button logic
    if canvas_result.image_data is not None and st.button("Analyze Whiteboard"):
        with st.spinner("Analyzing your drawing..."):
            # Convert canvas image data to PIL Image
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
            
            # Save image and send to analyzer
            image_path = "whiteboard_capture.png"
            image.save(image_path)
            result = st.session_state.analyzer.analyze_screen(image_path)
            st.success("Analysis Complete")
            st.markdown("### Analysis Result")
            st.write(result)
    
   

if __name__ == "__main__":
    main()