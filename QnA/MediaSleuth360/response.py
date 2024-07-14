import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()
# client = Groq(api_key=st.secrets["GROQ_API_KEY"])
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash",
                              generation_config=genai.GenerationConfig(temperature=0.1)
)

def generate_response(prompt: str, system: str) -> str:
    response = model.generate_content([system, prompt])
    # if error in response then display error message
    if "error" in response:
        return f"""[ERROR {response["error"]["type"]}]: {response["error"]["message"]}"""
    else:
        return response.text