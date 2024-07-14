import streamlit as st
from raw_data import generate_raw
import time


@st.cache_data()
def generate_transcript(file, filetype) -> str:
    # get file duration in HH:MM:SS format
    raw_data = generate_raw(file, filetype)

    # print("-------------------file-file--------------------------")
    # print(type(raw_data))
    # print("-------------------file--file-------------------------")
    # if size>25 MB it will return list of raw/transcriptions
    if type(raw_data) is not list:
        transcript = f"""filetype: {filetype}
Total Duration: {time.strftime('%H:%M:%S', time.gmtime(raw_data.duration))}
Language: {raw_data.language}
\n\nTranscript:\n"""
        for line in raw_data.segments:
            start_time = time.strftime('%H:%M:%S', time.gmtime(line['start']))
            end_time = time.strftime('%H:%M:%S', time.gmtime(line['end']))
            transcript += f"{start_time} - {end_time} : {line['text']}\n"
        return transcript
    else:
        raw_data = generate_raw(file, filetype)
        # print("================================================")
        # # print(len(raw_data))
        # print("================================================")
        transcript = f"filetype: {filetype}\n"

        total_duration = sum(
            segment['end'] - segment['start'] for chunk_result in raw_data for segment in chunk_result.segments)
        transcript += f"Total Duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}\n"

        for chunk_result in raw_data:
            transcript += f"Language: {chunk_result.language}\n\n"

        transcript += f"Transcript:\n"

        for chunk_result in raw_data:
            for line in chunk_result.segments:
                start_time = time.strftime('%H:%M:%S', time.gmtime(line['start']))
                end_time = time.strftime('%H:%M:%S', time.gmtime(line['end']))
                transcript += f"{start_time} - {end_time} : {line['text']}\n"
        return transcript