import os
import time
from pytube import YouTube
from moviepy.editor import VideoFileClip
from google.cloud import speech_v1p1beta1 as speech
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
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai_api_key = os.getenv("GOOGLE_API_KEY")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")  # Ensure this is set in your environment
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize LangChain with Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=genai_api_key)


# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e


# Function to transcribe audio using AssemblyAI
def transcribe_audio_assemblyai(audio_path):
    headers = {
        'authorization': assemblyai_api_key,
        'content-type': 'application/json'
    }

    # Upload the audio file
    upload_url = 'https://api.assemblyai.com/v2/upload'
    with open(audio_path, 'rb') as f:
        response = requests.post(upload_url, headers=headers, files={'file': f})
    audio_url = response.json()['upload_url']

    # Request transcription
    transcript_url = 'https://api.assemblyai.com/v2/transcript'
    transcript_request = {
        'audio_url': audio_url,
        'auto_chapters': True
    }
    response = requests.post(transcript_url, json=transcript_request, headers=headers)
    transcript_id = response.json()['id']

    # Wait for transcription to complete
    while True:
        response = requests.get(f'{transcript_url}/{transcript_id}', headers=headers)
        status = response.json()['status']
        if status == 'completed':
            break
        elif status == 'failed':
            raise Exception('Transcription failed')
        time.sleep(5)

    # Get the transcription and chapters
    transcription = response.json()['text']
    chapters = response.json()['chapters']
    return transcription, chapters


# Function to extract audio from video and convert to mono
def extract_audio(video_path, audio_path='audio.wav'):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(audio_path, format="wav")
    return audio_path


# Function to generate summary with timestamps
def generate_summary_with_timestamps(transcript_text, chapters):
    summary_prompt = PromptTemplate.from_template(
        "Summarize the following transcript and include important timestamps:\n\n{transcript}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run({"transcript": transcript_text})

    # Append timestamps to the summary
    summary_with_timestamps = summary + "\n\nTimestamps:\n"
    for chapter in chapters:
        start_time = time.strftime('%H:%M:%S', time.gmtime(chapter['start'] / 1000))
        end_time = time.strftime('%H:%M:%S', time.gmtime(chapter['end'] / 1000))
        summary_with_timestamps += f"{start_time} - {end_time}: {chapter['summary']}\n"

    return summary_with_timestamps


# Function to split transcript into documents
def split_transcript_into_documents(transcript_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = [Document(page_content=transcript_text)]
    split_documents = text_splitter.split_documents(documents)
    return split_documents


# Streamlit frontend
st.set_page_config(page_title="YouTube Summarizer and QnA", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("YouTube Summarizer and QnA")
youtube_link = st.text_input("Enter YouTube Video Link:")
uploaded_file = st.file_uploader("Upload a video or audio file", type=["mp4", "mp3", "wav"])

if youtube_link:
    video_id = youtube_link.split("=")[1]
    col1, col2, col3 = st.columns([1, 4, 1])  # Adjusted column widths to make the video area 20% wider
    with col2:
        st.video(youtube_link)

if uploaded_file:
    with open("uploaded_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    col1, col2, col3 = st.columns([1, 4, 1])  # Adjusted column widths to make the video area 20% wider
    with col2:
        st.video("uploaded_file")

if st.button("Get Detailed Notes"):
    if youtube_link:
        transcript_text = extract_transcript_details(youtube_link)
        chapters = []  # No chapters for YouTube transcript
    elif uploaded_file:
        audio_path = extract_audio("uploaded_file")
        transcript_text, chapters = transcribe_audio_assemblyai(audio_path)
    else:
        transcript_text = None

    if transcript_text:
        # Split transcript into documents
        documents = split_transcript_into_documents(transcript_text)
        st.session_state.documents = documents
        # Generate summary with timestamps
        summary = generate_summary_with_timestamps(transcript_text, chapters)
        st.session_state.summary = summary

if "summary" in st.session_state:
    st.markdown("## Summary with Timestamps:")
    st.write(st.session_state.summary)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
question = st.text_input("Ask a question about the video:")
if st.button("Ask Question"):
    if question:
        retriever = FAISS.from_documents(st.session_state.documents, embedding_model).as_retriever()
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template(
                "Answer the question based on the context: {context}")),
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_variable_name="context"
        )
        question_generator_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(
            "Generate a question based on the context: {context}"))
        retrieval_chain = ConversationalRetrievalChain(
            combine_docs_chain=combine_docs_chain,
            retriever=retriever,
            question_generator=question_generator_chain,
            return_source_documents=True
        )
        response = retrieval_chain({"question": question, "chat_history": []})
        st.markdown("## Answer:")
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["source_documents"]):
                st.write(doc.page_content)
                st.write("--------------------------------")