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
import base64
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

def initialize_session_state():
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
    
    # Add JavaScript for audio recording
    st.markdown("""
        <script>
        let mediaRecorder;
        let audioChunks = [];
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64Audio = reader.result.split(',')[1];
                        window.parent.postMessage({ type: 'audioData', data: base64Audio }, '*');
                    }
                };
                
                mediaRecorder.start();
                return true;
            } catch (err) {
                console.error('Error accessing microphone:', err);
                return false;
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
        
        window.addEventListener('message', function(event) {
            if (event.data.type === 'startRecording') {
                startRecording();
            } else if (event.data.type === 'stopRecording') {
                stopRecording();
            }
        });
        </script>
    """, unsafe_allow_html=True)
    
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
                    st.markdown("""
                        <script>
                        window.parent.postMessage({ type: 'startRecording' }, '*');
                        </script>
                    """, unsafe_allow_html=True)
                    st.rerun()
                else:
                    # Stop recording
                    st.session_state.is_recording = False
                    st.markdown("""
                        <script>
                        window.parent.postMessage({ type: 'stopRecording' }, '*');
                        </script>
                    """, unsafe_allow_html=True)
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