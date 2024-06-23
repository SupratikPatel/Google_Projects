import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configure Groq API
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LangChain with Groq API
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e

# Function to generate summary with timestamps
def generate_summary_with_timestamps(transcript_text):
    summary_prompt = PromptTemplate.from_template(
        "Summarize the following transcript and include important timestamps:\n\n{transcript}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run({"transcript": transcript_text})
    return summary

# Streamlit frontend
st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    if transcript_text:
        # Generate summary with timestamps
        summary = generate_summary_with_timestamps(transcript_text)
        st.markdown("## Summary with Timestamps:")
        st.write(summary)