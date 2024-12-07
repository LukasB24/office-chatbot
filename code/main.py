import postgres_handler
import mongo_handler
import pandas as pd
from file_scraper import extract_lines_with_two_speaker_changes
import ollama

if __name__ == '__main__':
    postgres_handler = postgres_handler.PostgresHandler()
    mongo_handler = mongo_handler.MongoHandler()

    data = pd.read_csv("the-office_lines.csv")
    test = extract_lines_with_two_speaker_changes(data.iloc[0:10])

    for dialogs in test:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=dialogs)
        embedding = response["embedding"]

    pass