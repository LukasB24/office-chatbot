class Document:
    def __init__(self, text, season, episode):
        self.text = text
        self.season = season
        self.episode = episode

def extract_lines_with_two_speaker_changes(df):
    result = []
    current_group = []
    previous_speaker = None
    speaker_changes = 0

    for index, row in df.iterrows():
        speaker = row['Character']
        line = row['Line']

        if previous_speaker and previous_speaker != speaker:
            speaker_changes += 1

        current_group.append(f"{speaker}: {line}")

        if speaker_changes == 4:
            result.append(" ".join(current_group))
            current_group = []
            speaker_changes = 0

        previous_speaker = speaker

    if current_group:
        result.append(" ".join(current_group))

    return result
