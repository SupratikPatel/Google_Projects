import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Vertex AI LLM
llm = VertexAI(model_name="gemini-pro")

# Streamlit UI
st.title("Video Summarizer and Q&A App")
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    # Load and chunk the video
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    result = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(result)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Store documents in ChromaDB
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a retriever chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Summarize video
    summary = qa({"query": "Summarize the video"})
    st.write("Video Summary:")
    st.write(summary["result"])

    # Q&A
    question = st.text_input("Ask a question about the video")
    if question:
        answer = qa({"query": question})
        st.write("Answer:")
        st.write(answer["result"])
