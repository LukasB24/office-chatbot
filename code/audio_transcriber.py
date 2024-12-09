import whisper
import os
import re

def extract_series_season_episode(filename: str) -> tuple[str, int, int]:
    match = re.search(r"(.+?)_S?(\d+)_E?(\d+)", filename, re.IGNORECASE)
    if match:
        series_name = match.group(1).replace("_", " ").title()
        season = int(match.group(2))
        episode = int(match.group(3))
        return series_name, season, episode
    else:
        raise ValueError("Could not extract series, season and episode from filename")


def transcribe_audio(video_file_path: str, language="de"):
    model = whisper.load_model("base")

    audio_file = "audio.mp3"
    os.system(f"ffmpeg -i {video_file_path} -q:a 0 -map a {audio_file}")

    result = model.transcribe(audio_file, language=language)

    os.remove(audio_file)

    return result['text']

def create_json_from_transcription(transcription: str, title: str):
    series_name, season, episode = extract_series_season_episode(title)

    result = {
        "title": series_name,
        "season": season,
        "episode": episode,
        "transcription": transcription
    }

    return result
