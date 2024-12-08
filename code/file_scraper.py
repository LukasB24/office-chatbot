import pandas as pd


class Document:
    def __init__(self, text: str, season: int, episode: int):
        self.text = text
        self.season = season
        self.episode = episode

def chunk_dynamically(df: pd.DataFrame) -> list[Document]:
    result = []
    current_group = []
    previous_speaker = None
    speaker_changes = 0
    season = None
    episode = None
    previous_episode = 1

    for index, row in df.iterrows():
        speaker = row['Character']
        line = row['Line']
        season = row['Season']
        episode = row['Episode_Number']

        if previous_speaker and previous_speaker != speaker:
            speaker_changes += 1

        if episode == previous_episode:
            current_group.append(f"{speaker}: {line}")

        if speaker_changes == 4 or episode != previous_episode:
            result.append(Document(" ".join(current_group), season, episode))
            current_group = []
            previous_episode = episode
            speaker_changes = 0

        previous_speaker = speaker

    if current_group:
        result.append(Document(" ".join(current_group), season, episode))

    return result