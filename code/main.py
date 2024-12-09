import postgres_handler
import mongo_handler
import pandas as pd
from minio_handler import MinioHandler
from audio_transcriber import transcribe_audio, create_json_from_transcription
from file_scraper import chunk_dynamically
import ollama

def extract_emotions_from_dialog(dialog: str):
    example = "{{'character': 'Michael', 'emotions': ['happy', 'tired']}, {'character': 'Pam', 'emotions': ['sad']}}"
    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"""
            How do the persons in in this conversation feel? \n 
            1. Please describe the emotions of each person in this conversation with exactly one word. \n 
            2. One person sometimes says multiple sentences but I want you to describe the emotion of this person in general with one word. 
            This means characters in this conversation should not appear twice in your answer. \n 
            3. Return the result as json like this but don't use exactly those values: {example} \n 
            4. Describe with one word like anger, uncomfortable, happy, sad, surprised or something else. \n 
            5. Don't provide whole sentences only the json. \n 
            6. Make sure that characters dont appear multiple times in the whole JSON. \n 
            7. follow the rules 1-7 ; \n 
            conversation: {dialog}. 
        """,
        options={"temperature": 0}
    )
    return output['response']

def insert_data():
    postgres = postgres_handler.PostgresHandler()
    mongo = mongo_handler.MongoHandler()

    data = pd.read_csv("the-office_lines.csv")
    chunks = chunk_dynamically(data.iloc[0:209])

    for document in chunks:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=document.text)
        embedding = response["embedding"]
        postgres_id = postgres.insert_data(embedding, document.text, document.episode, document.season)
        emotions = extract_emotions_from_dialog(document.text)

        if postgres_id != -1:
            mongo.insert_metadata({"postgres_id": postgres_id, "emotions": emotions})
            print(f"Inserted document with id {postgres_id} and emotions {emotions}")

if __name__ == '__main__':
    minio_handler = MinioHandler(
        endpoint="localhost:9000",
        username="minioadmin",
        password="minioadmin",
        bucket_name="office-transcripts"
    )

    filename = "the_office_2_19"

    transcript = transcribe_audio(filename + ".mp4")
    episode_json = create_json_from_transcription(transcript, filename + ".mp4")

    minio_handler.upload_json(filename + ".json", episode_json)
    #insert_data()

