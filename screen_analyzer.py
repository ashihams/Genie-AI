from google import genai
from google.genai import types
from PIL import Image
import os
from dotenv import load_dotenv


class ScreenAnalyzer:
    def __init__(self):
        """Initialize the ScreenAnalyzer with the GenAI client."""
        load_dotenv()
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )

    def analyze_screen(self, image_path):
        """Analyzes only the whiteboard content."""
        with Image.open(image_path) as img:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=["Analyze the mathematical expression in this image.", img],
                config=types.GenerateContentConfig(
                    system_instruction="Only return the computed result of the mathematical expression.",
                ),
            )
        return response.text
