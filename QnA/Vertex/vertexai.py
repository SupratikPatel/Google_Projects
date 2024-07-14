import os
from pytube import YouTube
from moviepy.editor import VideoFileClip
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment
import wave
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatPerplexity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from src.video_info import GetVideo
from src.model import Model
from src.prompt import Prompt
from src.misc import Misc
from src.timestamp_formatter import TimestampFormatter
from src.copy_module_edit import ModuleEditor
from st_copy_to_clipboard import st_copy_to_clipboard

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai_api_key = os.getenv("GOOGLE_API_KEY")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")  # Ensure this is set in your environment
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")

# Initialize LangChain with Groq API
llm1 = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
# Initialize LangChain with Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=genai_api_key)
llm2 = ChatPerplexity(model="llama-3-sonar-large-32k-online")


class AIVideoSummarizer:
    def __init__(self):
        self.youtube_url = None
        self.video_id = None
        self.video_title = None
        self.video_transcript = None
        self.video_transcript_time = None
        self.summary = None
        self.time_stamps = None
        self.transcript = None
        self.model_name = None
        self.col1 = None
        self.col2 = None
        self.col3 = None
        self.model_env_checker = []
        load_dotenv()

    def get_youtube_info(self):
        self.youtube_url = st.text_input("Enter YouTube Video Link")

        if os.getenv("GOOGLE_GEMINI_API_KEY"):
            self.model_env_checker.append("Gemini")
        if os.getenv("OPENAI_CHATGPT_API_KEY"):
            self.model_env_checker.append("ChatGPT")
        if not self.model_env_checker:
            st.warning('Error while loading the API keys from environment.', icon="⚠️")

        with self.col2:
            if self.model_env_checker:
                with st.container():
                    self.model_name = st.selectbox(
                        'Select the model',
                        self.model_env_checker)

                    def switch(model_name):
                        if model_name == "Gemini":
                            st.columns(3)[1].image("https://i.imgur.com/w9izNH5.png", use_column_width=True)
                        elif model_name == "ChatGPT":
                            st.columns(3)[1].image("https://i.imgur.com/Sr9e9ZC.png", use_column_width=True)

                    switch(self.model_name)

        if self.youtube_url:
            self.video_id = GetVideo.Id(self.youtube_url)
            if self.video_id is None:
                st.write("**Error**")
                st.image("https://i.imgur.com/KWFtgxB.png", use_column_width=True)
                st.stop()
            self.video_title = GetVideo.title(self.youtube_url)
            st.write(f"**{self.video_title}**")
            st.image(f"http://img.youtube.com/vi/{self.video_id}/0.jpg", use_column_width=True)

    def generate_summary(self):
        if st.button(":rainbow[**Get Summary**]"):
            self.video_transcript = GetVideo.transcript(self.youtube_url)
            if self.model_name == "Gemini":
                self.summary = Model.google_gemini(transcript=self.video_transcript, prompt=Prompt.prompt1())
            elif self.model_name == "ChatGPT":
                self.summary = Model.openai_chatgpt(transcript=self.video_transcript, prompt=Prompt.prompt1())
            st.markdown("## Summary:")
            st.write(self.summary)
            st_copy_to_clipboard(self.summary)

    def generate_time_stamps(self):
        if st.button(":rainbow[**Get Timestamps**]"):
            self.video_transcript_time = GetVideo.transcript_time(self.youtube_url)
            youtube_url_full = f"https://youtube.com/watch?v={self.video_id}"
            if self.model_name == "Gemini":
                self.time_stamps = Model.google_gemini(self.video_transcript_time, Prompt.prompt1(ID='timestamp'),
                                                       extra=youtube_url_full)
            elif self.model_name == "ChatGPT":
                self.time_stamps = Model.openai_chatgpt(self.video_transcript_time, Prompt.prompt1(ID='timestamp'),
                                                        extra=youtube_url_full)
            st.markdown("## Timestamps:")
            st.markdown(self.time_stamps)
            cp_text = TimestampFormatter.format(self.time_stamps)
            st_copy_to_clipboard(cp_text)

    def generate_transcript(self):
        if st.button("Get Transcript"):
            self.video_transcript = GetVideo.transcript(self.youtube_url)
            self.transcript = self.video_transcript
            st.markdown("## Transcript:")
            st.download_button(label="Download as text file", data=self.transcript,
                               file_name=f"Transcript - {self.video_title}")
            st.write(self.transcript)
            st_copy_to_clipboard(self.transcript)

    def run(self):
        st.set_page_config(page_title="AI Video Summarizer", page_icon="chart_with_upwards_trend", layout="wide")
        st.title("AI Video Summarizer")
        editor = ModuleEditor('st_copy_to_clipboard')
        editor.modify_frontend_files()

        self.col1, self.col2, self.col3 = st.columns(3)

        with self.col1:
            self.get_youtube_info()

        ran_loader = Misc.loaderx()
        n, loader = ran_loader[0], ran_loader[1]

        with self.col3:
            mode = st.radio(
                "What do you want to generate for this video?",
                [":rainbow[**AI Summary**]", ":rainbow[**AI Timestamps**]", "**Transcript**"],
                index=0)
            if mode == ":rainbow[**AI Summary**]":
                with st.spinner(loader[n]):
                    self.generate_summary()
            elif mode == ":rainbow[**AI Timestamps**]":
                with st.spinner(loader[n]):
                    self.generate_time_stamps()
            else:
                with st.spinner(loader[0]):
                    self.generate_transcript()

        st.write(Misc.footer(), unsafe_allow_html=True)


if __name__ == "__main__":
    app = AIVideoSummarizer()
    app.run()