Youtube video summarizer and QnA using Langchain , Groq, FAISS, GoogleAiEmbeddings and streamlit

Any offline video/audio summariser using Gemini 



1. AnyVideo_QnA has 2 apps namely 
(a) Gemini (stores the converted transcripts and audio file on google cloud bucket) and
   (b) Gemini_OfflineDataStore (stores the transcripts and audio file locally for faster data processing)


2. YTLink_QnA has 2 apps namely
(a) QnA that is summarizer with QnA feature using GroQ llm using any YT link
(b) Summarizer which is simple summarizer using YT link and google llm