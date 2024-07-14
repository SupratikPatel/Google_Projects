import os
import math
# import eyed3
import logging
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_SIZE_MB = 25

def calculate_segment_duration_and_num_segments(duration_seconds, overlap_seconds, max_size, bitrate_kbps):
    """Calculate the duration and number of segments for an audio file."""
    seconds_for_max_size = (max_size * 8 * 1024) / bitrate_kbps
    num_segments = max(2, int(duration_seconds / seconds_for_max_size) + 1)
    total_overlap = (num_segments - 1) * overlap_seconds
    actual_playable_duration = (duration_seconds - total_overlap) / num_segments
    return num_segments, actual_playable_duration + overlap_seconds

def construct_file_names(path_to_mp3, num_segments):
    """Construct new file names for the segments of an audio file."""
    directory = os.path.dirname(path_to_mp3)
    base_name = os.path.splitext(os.path.basename(path_to_mp3))[0]
    padding = max(1, int(math.ceil(math.log10(num_segments))))
    new_names = [os.path.join(directory, f"{base_name}_{str(i).zfill(padding)}.mp3") for i in range(1, num_segments + 1)]
    return new_names

def split_mp3(path_to_mp3, overlap_seconds, max_size=MAX_SIZE_MB):
    """Split an mp3 file into segments."""
    # if not os.path.exists(path_to_mp3):
    #     raise ValueError(f"File {path_to_mp3} does not exist.")
    audio_file = path_to_mp3
    if audio_file is None:
        raise ValueError(f"File {path_to_mp3} is not a valid mp3 file.")
    duration_seconds = audio_file.info.time_secs
    bitrate_kbps = audio_file.info.bit_rate[1]
    file_size_MB = os.path.getsize(path_to_mp3) / (1024 * 1024)
    if file_size_MB < max_size:
        logging.info("File is less than maximum size, no action taken.")
        return path_to_mp3
    num_segments, segment_duration = calculate_segment_duration_and_num_segments(duration_seconds, overlap_seconds, max_size, bitrate_kbps)
    new_file_names = construct_file_names(path_to_mp3, num_segments)
    original_audio = AudioSegment.from_mp3(path_to_mp3)
    start = 0
    for i in range(num_segments):
        if i == num_segments - 1:
            segment = original_audio[start:]
        else:
            end = start + segment_duration * 1000
            segment = original_audio[start:int(end)]
        segment.export(new_file_names[i], format="mp3")
        start += (segment_duration - overlap_seconds) * 1000
    logging.info(f"Split into {num_segments} sub-files.")
    return new_file_names