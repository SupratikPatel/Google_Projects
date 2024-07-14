import streamlit as st
from response import generate_response
from transcribe import generate_transcript

@st.cache_data()
def generate_summary(file, filetype) -> str:
    transcript = generate_transcript(file, filetype)

    system_prompt = f"""You are an expert media analyst. Your task is \
to create a structured summary of the given {filetype} transcript. The \
summary should include key chapters or topics discussed, each with \
corresponding timestamps. Follow the exact format provided below:

<format-start>
start_time - end_time: Topic 1 summary.\n
start_time - end_time: Topic 2 summary.\n
start_time - end_time: Topic 3 summary.\n
...
<format-end>

Note: Ensure that the summary is concise and focuses on the main points. \
Strictly adhere to the format and do not include any additional information \
or commentary. Summary timestamps cant exceed total duration of the {filetype}. \
Do not include the start and end tags in the summary itself.
"""

    user_prompt = f"""Generate a structured summary from the {filetype} \
transcript provided below, identifying key chapters or topics discussed \
along with their timestamps:\n\n{transcript}"""

    return generate_response(system_prompt, user_prompt)