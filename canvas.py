import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")

st.title("Minimal Whiteboard Test")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",  # No fill color
    stroke_width=3,
    stroke_color="#000000",
    background_color="#EEEEEE",
    height=500,
    width=800,
    drawing_mode="freedraw",
    key="test_canvas",
)

if canvas_result.image_data is not None:
    st.write("Canvas detected!") 