# Must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Genie AI - Omni Tutor",
    page_icon="🧠",
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
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import custom modules
from audio_utils import AudioRecorder, transcribe_audio, text_to_speech
from ai_utils import get_ai_response, analyze_drawing, generate_practice_problem
from whiteboard import Whiteboard

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE")

# Initialize clients
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

# Constants
LANGUAGES = {
    "English": {"code": "en-US", "voice": "en-US-Studio-O"},
    "Hindi": {"code": "hi-IN", "voice": "hi-IN-Standard-A"},
    "Spanish": {"code": "es-ES", "voice": "es-ES-Standard-A"},
    "French": {"code": "fr-FR", "voice": "fr-FR-Standard-A"},
    "German": {"code": "de-DE", "voice": "de-DE-Standard-A"},
    "Japanese": {"code": "ja-JP", "voice": "ja-JP-Standard-A"},
    "Chinese": {"code": "zh-CN", "voice": "zh-CN-Standard-A"}
}

SUBJECTS = {
    "Mathematics": ["Algebra", "Geometry", "Calculus", "Statistics", "Trigonometry"],
    "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Quantum Physics"],
    "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Biochemistry"],
    "Biology": ["Cell Biology", "Genetics", "Ecology", "Anatomy", "Evolution"],
    "Computer Science": ["Programming", "Data Structures", "Algorithms", "Machine Learning", "Web Development"]
}

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

def generate_solution_response(problem, selected_language, model):
    if not problem:
        return "No problem available to generate a solution for."
    
    prompt = f"Provide a detailed, step-by-step solution for the following problem, appropriate for a student learning {st.session_state.current_subject} at a {st.session_state.difficulty_level} level:\n\nProblem: {problem}"
    response = get_ai_response(prompt, selected_language, model)
    return response

def initialize_session_state():
    """Initialize all session state variables"""
    if "recorder" not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if "whiteboard" not in st.session_state:
        st.session_state.whiteboard = Whiteboard()
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = "Mathematics"
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = "Algebra"
    if "difficulty_level" not in st.session_state:
        st.session_state.difficulty_level = "Intermediate"
    if "current_problem" not in st.session_state:
        st.session_state.current_problem = ""
    if "selected_color" not in st.session_state:
        st.session_state.selected_color = "#000000"
    if "stroke_width" not in st.session_state:
        st.session_state.stroke_width = 5
    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = "freedraw"
    if "clear_canvas_counter" not in st.session_state:
        st.session_state.clear_canvas_counter = 0

def main():
    initialize_session_state()
    
    # Sidebar for voice assistant and settings
    with st.sidebar:
        st.markdown("### 🧠 Omni Tutor Settings")
        
        # Subject and topic selection
        selected_subject = st.selectbox(
            "📚 Subject",
            options=list(SUBJECTS.keys()),
            index=list(SUBJECTS.keys()).index(st.session_state.current_subject)
        )
        
        if selected_subject != st.session_state.current_subject:
            st.session_state.current_subject = selected_subject
            st.session_state.current_topic = SUBJECTS[selected_subject][0]
        
        selected_topic = st.selectbox(
            "📖 Topic",
            options=SUBJECTS[selected_subject],
            index=SUBJECTS[selected_subject].index(st.session_state.current_topic) if st.session_state.current_topic in SUBJECTS[selected_subject] else 0
        )
        st.session_state.current_topic = selected_topic
        
        # Difficulty level
        difficulty = st.selectbox(
            "🎯 Difficulty Level",
            options=DIFFICULTY_LEVELS,
            index=DIFFICULTY_LEVELS.index(st.session_state.difficulty_level)
        )
        st.session_state.difficulty_level = difficulty
        
        # Language selection
        selected_language = st.selectbox(
            "🌍 Select Language",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )
        
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
        
        st.markdown("---")
        
        # Voice Assistant Section
        st.markdown("### 🎤 Voice Assistant")
        
        # Google Assistant-like button styling
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
            div.stButton > button:active {
                background-color: #ea4335;
            }
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            button_label = "🎤" if not st.session_state.is_recording else "⏹️"
            if st.button(button_label, key="sidebar_toggle_recording"):
                if not st.session_state.is_recording:
                    # Start recording
                    st.session_state.is_recording = True
                    st.session_state.recorder.start_recording()
                    st.rerun()
                else:
                    # Stop recording and process
                    st.session_state.is_recording = False
                    audio_data = st.session_state.recorder.stop_recording()
                    
                    if audio_data:
                        # Get language settings
                        lang_code = LANGUAGES[st.session_state.selected_language]["code"]
                        voice_name = LANGUAGES[st.session_state.selected_language]["voice"]
                        
                        # Convert speech to text
                        text = transcribe_audio(audio_data, lang_code, speech_client)
                        
                        if text and text != "No speech detected":
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                "role": "student",
                                "content": text,
                                "timestamp": datetime.now()
                            })
                            
                            # Get tutoring context
                            context = f"Subject: {st.session_state.current_subject}, Topic: {st.session_state.current_topic}, Level: {st.session_state.difficulty_level}"
                            if st.session_state.current_problem:
                                context += f"\nCurrent Problem: {st.session_state.current_problem[:200]}..."
                            
                            # Get AI response
                            response = get_ai_response(text, st.session_state.selected_language, model)
                            st.session_state.last_response = response
                            
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                "role": "tutor",
                                "content": response,
                                "timestamp": datetime.now()
                            })
                            
                            # Generate speech
                            audio_file = text_to_speech(response, lang_code, voice_name, tts_client)
                            if audio_file:
                                st.session_state.audio_file = audio_file
                    st.rerun()
        
        # Status indicator
        status_text = "🔴 Listening..." if st.session_state.is_recording else "🎤 Tap to speak"
        st.markdown(f"<div style='text-align: center; margin-top: 10px; font-size: 14px; color: #666;'>{status_text}</div>", unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show last response and auto-play audio
        if st.session_state.last_response and st.session_state.audio_file:
            st.markdown("**🧠 Tutor Response:**")
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0; font-size: 14px;'>{st.session_state.last_response}</div>", unsafe_allow_html=True)
            st.audio(st.session_state.audio_file, format="audio/mp3", autoplay=True)
    
    # Main content area
    st.markdown("<h1 style='text-align: center; color: #1E88E5; margin-bottom: 10px;'>🧠 Genie AI - Omni Tutor</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: #666; margin-bottom: 30px;'>Learning {st.session_state.current_subject} • {st.session_state.current_topic} • {st.session_state.difficulty_level}</h3>", unsafe_allow_html=True)
    
    # Main area tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat with Tutor", "📝 Interactive Whiteboard", "📊 Progress & Stats"])
    
    with tab1:
        st.markdown("### 💬 Tutoring Conversation")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history[-15:]):  # Show last 15 messages
                if message["role"] == "student":
                    st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 15px; margin: 10px 0; margin-left: 50px; border-left: 4px solid #2196f3;'>
                        <strong>👤 You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 15px; margin: 10px 0; margin-right: 50px; border-left: 4px solid #4caf50;'>
                        <strong>🧠 Tutor:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Text input for typing
        st.markdown("---")
        user_input = st.text_area("💭 Type your question or share your work:", height=100, placeholder="Ask me anything about the topic, share your solution, or request help with a specific concept...")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            send_button = st.button("Send 📤", type="primary", use_container_width=True)
        
        if send_button and user_input:
            # Add to conversation history
            st.session_state.conversation_history.append({
                "role": "student",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get tutoring context
            context = f"Subject: {st.session_state.current_subject}, Topic: {st.session_state.current_topic}, Level: {st.session_state.difficulty_level}"
            if st.session_state.current_problem:
                context += f"\nCurrent Problem: {st.session_state.current_problem[:200]}..."
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                response = get_ai_response(user_input, st.session_state.selected_language, model)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "role": "tutor",
                "content": response,
                "timestamp": datetime.now()
            })
            
            st.rerun()
    
    with tab2:
        # Header and Generate Problem button
        col_title, col_button = st.columns([4, 1]) # Adjust column ratios as needed
        with col_title:
            st.markdown("### 📝 Interactive Whiteboard")
        with col_button:
            # Add custom CSS for button alignment
            st.markdown("""
            <style>
            div[data-testid="stColumn"] > div > div > button {
                margin-top: 20px; /* Adjust this value to align with the title */
                float: right; /* Pushes the button to the right within its column */
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("📝 Generate New Problem", type="primary", use_container_width=False, key="generate_problem_button"): # Set use_container_width to False for better control
                with st.spinner("Creating a personalized problem..."):
                    problem = generate_practice_problem(
                        st.session_state.current_subject,
                        st.session_state.current_topic,
                        st.session_state.difficulty_level,
                        model
                    )
                    st.session_state.current_problem = problem
                    st.session_state.clear_canvas_counter += 1
                    st.rerun()

        # Display current problem if exists
        if st.session_state.current_problem:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 4px solid #4285F4;'>
                <strong>📝 Current Problem:</strong><br><br>
                {st.session_state.current_problem.split('SOLUTION_STEPS')[0].strip()}
            </div>
            """, unsafe_allow_html=True)

        # Whiteboard controls
        st.markdown("### 🎨 Drawing Controls")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            drawing_mode = st.selectbox(
                "Drawing Tool", 
                ["freedraw", "line", "rect", "circle"],
                key="drawing_tool"
            )
        
        with col2:
            if st.checkbox("Eraser Mode", key="eraser_mode"):
                stroke_color = "#FFFFFF"  # White for erasing
                st.markdown("**🗑️ Eraser Active**")
            else:
                stroke_color = st.color_picker("Pen Color", "#000000", key="pen_color")
        
        with col3:
            stroke_width = st.slider("Brush Size", 1, 20, 5, key="brush_size")
        
        with col4:
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                st.session_state.clear_canvas_counter += 1
                st.rerun()

        # Canvas container with better styling
        st.markdown("### ✏️ Draw Your Solution Below")
        
        # Enhanced CSS for canvas visibility
        st.markdown("""
        <style>
        .stCanvas > div {
            border: 3px solid #4285F4 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3) !important;
            background-color: white !important;
            margin: 20px 0 !important;
        }
        .stCanvas canvas {
            border-radius: 8px !important;
            background-color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a container for the canvas
        canvas_container = st.container()
        
        with canvas_container:
            # Main canvas with improved settings
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color="#FFFFFF",  # White background
                    background_image=None,
                    update_streamlit=True,
                    height=800,  # Fixed height
                    width=900,  # Increased width
                    drawing_mode=drawing_mode,
                    point_display_radius=3,
                    key=f"main_canvas_{st.session_state.clear_canvas_counter}",
                    display_toolbar=False,  # Hide default toolbar since we have custom controls
                )
                
            except Exception as e:
                st.error(f"Canvas error: {str(e)}")
                st.markdown("""
                **Troubleshooting Tips:**
                1. Make sure `streamlit-drawable-canvas` is installed: `pip install streamlit-drawable-canvas`
                2. Refresh the page if the canvas doesn't appear
                3. Try clearing your browser cache
                """)
                canvas_result = None

        # Analysis section - only show if canvas has content
        st.markdown("---")
        
        if canvas_result is not None and canvas_result.image_data is not None:
            # Check if there's actual drawing (not just white canvas)
            image_array = canvas_result.image_data
            has_drawing = not np.all(image_array == 255)  # Check if not all white
            
            if has_drawing:
                col_analyze, col_solution = st.columns(2)
                with col_analyze:
                    if st.button("🔍 Analyze", type="primary", use_container_width=True):
                        with st.spinner("🤔 Analyzing your solution..."):
                            try:
                                # Convert canvas image data to PIL Image
                                image = Image.fromarray(image_array.astype(np.uint8))
                                
                                # Save image temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                                    image.save(temp_file.name)
                                    temp_file_path = temp_file.name
                                    
                                    # Get tutoring context
                                    context = f"Subject: {st.session_state.current_subject}, Topic: {st.session_state.current_topic}, Level: {st.session_state.difficulty_level}"
                                    if st.session_state.current_problem:
                                        context += f"\nCurrent Problem: {st.session_state.current_problem}"
                                    
                                    # Analyze the drawing
                                    result = analyze_drawing(temp_file_path, context, model)
                                    
                                    if result:
                                        # Get language settings for voice
                                        lang_code = LANGUAGES[st.session_state.selected_language]["code"]
                                        voice_name = LANGUAGES[st.session_state.selected_language]["voice"]
                                        
                                        # Generate voice feedback
                                        audio_file = text_to_speech(result, lang_code, voice_name, tts_client)
                                        if audio_file:
                                            st.session_state.audio_file = audio_file
                                        
                                        # Add to conversation history
                                        st.session_state.conversation_history.append({
                                            "role": "student",
                                            "content": "I've drawn my solution on the whiteboard.",
                                            "timestamp": datetime.now()
                                        })
                                        
                                        st.session_state.conversation_history.append({
                                            "role": "tutor",
                                            "content": f"📝 **Drawing Analysis:**\n\n{result}",
                                            "timestamp": datetime.now()
                                        })
                                        
                                        # Display analysis result
                                        st.success("✅ Analysis Complete!")
                                        st.markdown(f"""
                                        <div style='background-color: #e8f5e8; padding: 20px; border-radius: 15px; border-left: 4px solid #4caf50; margin: 20px 0;'>
                                            <strong>🧠 AI Tutor Feedback:</strong><br><br>
                                            {result}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Auto-play audio feedback
                                        if st.session_state.audio_file:
                                            st.audio(st.session_state.audio_file, format="audio/mp3", autoplay=True)
                                    
                                    # Clean up temp file
                                    if os.path.exists(temp_file_path):
                                        os.unlink(temp_file_path)
                            
                            except Exception as e:
                                st.error(f"Error analyzing drawing: {str(e)}")
                                st.markdown("Please try drawing again or check your internet connection.")
                with col_solution:
                    if st.button("💡 Generate Solution", type="secondary", use_container_width=True):
                        with st.spinner("✍️ Generating solution..."):
                            solution = generate_solution_response(
                                st.session_state.current_problem,
                                st.session_state.selected_language,
                                model
                            )
                            if solution:
                                st.session_state.conversation_history.append({
                                    "role": "tutor",
                                    "content": f"Here's a step-by-step solution to the problem:\n\n{solution}",
                                    "timestamp": datetime.now()
                                })
                                # Generate speech for the solution
                                lang_code = LANGUAGES[st.session_state.selected_language]["code"]
                                voice_name = LANGUAGES[st.session_state.selected_language]["voice"]
                                audio_file = text_to_speech(solution, lang_code, voice_name, tts_client)
                                if audio_file:
                                    st.session_state.audio_file = audio_file
                            st.rerun()
            else:
                st.info("🎨 Draw something on the canvas above to enable analysis!")
        else:
            st.info("🎨 The canvas will appear here. If you don't see it, try refreshing the page.")
    
    with tab3:
        st.markdown("### 📊 Your Learning Journey")
        
        # Session statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            questions_asked = len([msg for msg in st.session_state.conversation_history if msg["role"] == "student"])
            st.metric("❓ Questions Asked", questions_asked)
        
        with col2:
            responses_given = len([msg for msg in st.session_state.conversation_history if msg["role"] == "tutor"])
            st.metric("💬 Tutor Responses", responses_given)
        
        with col3:
            if st.session_state.conversation_history:
                session_time = datetime.now() - st.session_state.conversation_history[0]["timestamp"]
                minutes = session_time.seconds // 60
                st.metric("⏱️ Session Time", f"{minutes} min")
            else:
                st.metric("⏱️ Session Time", "0 min")
        
        with col4:
            current_streak = len(st.session_state.conversation_history) // 2  # Rough estimate
            st.metric("🔥 Interaction Streak", current_streak)
        
        # Subject focus
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Current Focus")
            st.info(f"""
            **Subject:** {st.session_state.current_subject}  
            **Topic:** {st.session_state.current_topic}  
            **Level:** {st.session_state.difficulty_level}
            """)
        
        with col2:
            st.markdown("#### 🎯 Session Summary")
            if st.session_state.conversation_history:
                recent_topics = [msg['content'][:50] + "..." for msg in st.session_state.conversation_history[-3:] if msg['role'] == 'student']
                if recent_topics:
                    st.write("**Recent questions:**")
                    for topic in recent_topics:
                        st.write(f"• {topic}")
                else:
                    st.write("Start asking questions to see your progress!")
            else:
                st.write("Your learning session is just beginning! 🌟")
        
        # Motivational section
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
            <h2>🌟 Keep Up the Great Work!</h2>
            <p style='font-size: 18px; margin: 15px 0;'>Every question you ask brings you closer to mastery.</p>
            <p style='font-size: 16px;'>I'm here to guide you through every step of your learning journey!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()