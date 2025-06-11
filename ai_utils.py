import streamlit as st
import google.generativeai as genai
from PIL import Image

def get_ai_response(text, language="English", model=None):
    """Get response from Gemini"""
    if not model:
        return "AI model not available. Please check your GOOGLE_API_KEY."
    
    try:
        # Enhanced tutoring prompt
        tutor_prompt = f"""
        You are an expert, patient, and encouraging tutor like Khan Academy's Omni Math Tutor. 

        Student question/response: {text}
        Language: {language}

        Respond as a personalized tutor:
        1. Be encouraging and supportive
        2. Provide step-by-step guidance when needed
        3. Ask follow-up questions to check understanding
        4. Give hints rather than direct answers
        5. Adapt to the student's level
        6. Use analogies and examples
        7. Celebrate progress and learning

        {"Respond in " + language if language != "English" else ""}
        """
        
        # Generate response with specific configuration
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.7,
        )
        
        response = model.generate_content(
            tutor_prompt,
            generation_config=generation_config
        )
        
        # Check if response was blocked or empty
        if not response.text:
            return "I'm here to help you learn! Could you rephrase your question?"
        
        return response.text.strip()
        
    except Exception as e:
        return f"I'm experiencing some technical difficulties, but I'm still here to help you learn! Error: {str(e)}"

def analyze_drawing(image_path, context="", model=None):
    """Analyze mathematical drawings and provide tutoring feedback"""
    if not model:
        return "AI model not available"
    
    try:
        # Upload image to Gemini
        image = Image.open(image_path)
        
        analysis_prompt = f"""
        You are an expert math tutor analyzing a student's work on a whiteboard/paper.
        
        Context: {context}
        
        Please analyze this image and provide tutoring feedback:
        1. What mathematical concepts or problems do you see?
        2. Are there any errors or misconceptions?
        3. What's done well?
        4. What guidance would help the student?
        5. What questions should I ask to check their understanding?
        
        Be encouraging and specific in your feedback.
        """
        
        response = model.generate_content([analysis_prompt, image])
        return response.text.strip()
        
    except Exception as e:
        return f"Error analyzing drawing: {str(e)}"

def generate_practice_problem(subject, topic, difficulty, model=None):
    """Generate practice problems"""
    if not model:
        return "AI model not available"
    
    try:
        problem_prompt = f"""
        Create a {difficulty} level practice problem for {subject} - {topic}.
        
        Provide the problem in this format:
        PROBLEM: [Clear problem statement]
        SOLUTION_STEPS: [Step-by-step solution]
        KEY_CONCEPTS: [Main concepts being tested]
        
        Make it engaging and practical.
        """
        
        response = model.generate_content(problem_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating problem: {str(e)}" 