import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile

class Whiteboard:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize whiteboard session state variables"""
        if "drawing_mode" not in st.session_state:
            st.session_state.drawing_mode = "freedraw"
        if "selected_color" not in st.session_state:
            st.session_state.selected_color = "#000000"
        if "stroke_width" not in st.session_state:
            st.session_state.stroke_width = 3
    
    def render(self, width=800, height=500):
        """Render the whiteboard component"""
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
            width=width,
            height=height,
            drawing_mode=drawing_mode,
            key="whiteboard_canvas"
        )
        
        return canvas_result
    
    def get_image(self, canvas_result):
        """Get the current canvas image as PIL Image"""
        if canvas_result.image_data is not None:
            return Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
        return None
    
    def save_image(self, canvas_result, filepath):
        """Save the current canvas image to a file"""
        image = self.get_image(canvas_result)
        if image:
            image.save(filepath)
            return True
        return False
    
    def clear(self):
        """Clear the whiteboard"""
        st.session_state.selected_color = "#FFFFFF"
        st.session_state.stroke_width = 50
        st.rerun() 