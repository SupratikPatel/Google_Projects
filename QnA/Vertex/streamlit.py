import streamlit as st
from vertexai import summarize_video, answer_question

st.title("Video Summarizer and Q&A App")
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    summary = summarize_video(video_url)
    st.write("Video Summary:")
    st.write(summary)

    question = st.text_input("Ask a question about the video")
    if question:
        answer = answer_question(video_url, question)
        st.write("Answer:")
        st.write(answer)