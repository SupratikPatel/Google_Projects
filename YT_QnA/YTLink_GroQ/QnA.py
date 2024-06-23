import os
from pytube import YouTube
from google.cloud import speech_v1p1beta1 as speech
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

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

if youtube_link:
    video_id = youtube_link.split("=")[1]
    col1, col2, col3 = st.columns([1, 4, 1])  # Adjusted column widths to make the video area 20% wider
    with col2:
        st.video(youtube_link)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    if transcript_text:
        # Split transcript into documents
        documents = split_transcript_into_documents(transcript_text)
        st.session_state.documents = documents
        # Generate summary with timestamps
        summary = generate_summary_with_timestamps(transcript_text)
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
            llm_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Answer the question based on the context: {context}")),
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_variable_name="context"
        )
        question_generator_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("Generate a question based on the context: {context}"))
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