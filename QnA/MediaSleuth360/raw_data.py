import streamlit as st
from groq import Groq
from moviepy.editor import VideoFileClip
import tempfile
import os
from pydub import AudioSegment
from io import BytesIO

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(file.getbuffer())
        temp_video_path = temp_video.name

    # Convert video to audio using MoviePy
    video_clip = VideoFileClip(temp_video_path)

    # if video_clip.audio is None:
    #         raise ValueError("The video file does not contain an audio track.")
    # else:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        temp_audio_path = temp_audio.name
        video_clip.audio.write_audiofile(temp_audio_path)

    # Remove the temporary video file
    os.remove(temp_video_path)

    # Open the converted audio file
    with open(temp_audio_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()

    return (temp_audio_path, audio_bytes)


@st.cache_data()
def generate_raw(file, filetype):
    # If the file is a video, convert it to audio first
    file_size = file.size / (1024 ** 2)

    if filetype.startswith('video/'):
        file = extract_audio(file)
        file_size = len(file[1]) / (1024 ** 2)

    # print("-------------------file---------------------------")
    # print(file)
    # print(type(file))
    # print(file_size)
    # print("--------------------file--------------------------")

    if file_size < 25.0:
        # Generate the raw data
        raw = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3",
            response_format="verbose_json"
        )

        # print("-------------------raw---------------------------")
        # # print(raw)
        # print("--------------------raw--------------------------")

        # Remove the temporary audio file
        if filetype.startswith('video/'):
            os.remove(file[0])

        return raw

    else:
        # Split the audio into chunks
        chunks = split_audio(file, chunk_duration_minutes=20)

        raw_results = []
        total_duration = 0

        # Transcribe the first chunk without any prompt
        first_chunk = chunks[0]
        firstraw = client.audio.transcriptions.create(
            file=open(first_chunk, "rb"),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        raw_results.append(firstraw)

        # print("------------------------------8888888888888----------------------------------------------------")
        # # print(raw_results.)
        # print("------------------------------8888888888888----------------------------------------------------")

        total_duration += [line['end'] for line in firstraw.segments][-1]
        os.remove(first_chunk)

        # Transcribe the remaining chunks with the previous transcription as a prompt and start time
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            prompt = " ".join([segment['text'] for segment in raw_results[-1].segments][-3:])
            start_time = [segment['end'] for segment in raw_results[-1].segments][-1]
            # print("----------------------------------------------------------------------------------")
            # print(len(raw_results[-1].segments))
            # print("----------------------------------------------------------------------------------")
            raw = client.audio.transcriptions.create(
                file=open(chunk, "rb"),
                model="whisper-large-v3",
                response_format="verbose_json",
                prompt=prompt
            )

            adjusted_raw = adjust_timestamps(raw, start_time)
            raw_results.append(adjusted_raw)
            total_duration += [line['end'] for line in raw.segments][-1]

            # os.remove(first_chunk)
            os.remove(chunk)

        # Remove the temporary audio file if it was converted from video
        if filetype.startswith('video/'):
            os.remove(file)

        return raw_results


def split_audio(file, chunk_duration_minutes=10):
    # Read the content of the uploaded file as bytes
    file_contents = file.getvalue()

    # Convert the bytes to a BytesIO object
    bytes_io = BytesIO(file_contents)
    audio = AudioSegment.from_file(bytes_io)
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    chunks = []

    for i, chunk in enumerate(audio[::chunk_duration_ms]):
        chunk_name = f"chunk_{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        chunks.append(chunk_name)
    # print("----------------------------------------------")
    # print(chunks)
    # print("----------------------------------------------")
    return chunks


def adjust_timestamps(chunk_result, start_time):
    for segment in chunk_result.segments:
        segment['start'] += start_time
        segment['end'] += start_time
    return chunk_result

