import postgres_handler
import mongo_handler
import pandas as pd
from file_scraper import extract_lines_with_two_speaker_changes
import ollama

def extract_emotions_from_dialog(dialog: str = None):
    statement = """Michael:  I’ve, uh, I’ve been at Dunder Mifflin for 12 years, the last four as Regional Manager. If you want to come through here… See we have the entire floor. So this is my kingdom, as far as the eye can see. This is our receptionist, Pam. Pam! Pam-Pam! Pam Beesly. Pam has been with us for…  forever. Right, Pam?
                 Pam: Well. I don’t know."
                 Michael: If you think she’s cute now, you should have seen her a couple of years ago. [growls] 
                 Pam: What?"""

    if not dialog:
        dialog = statement

    example = "{{'character': 'Michael', 'emotions': ['happy', 'tired']}, {'character': 'Pam', 'emotions': ['sad']}}"
    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"How do the persons in in this conversation feel? \n 1. Please describe the emotions of each person in this conversation with exactly one word. \n 2. One person sometimes says multiple sentences but I want you to describe the emotion of this person in general with one word. This means characters in this conversation should not appear twice in your answer. \n 3. Return the result as json like this but don't use exactly those values: {example} \n 4. Describe with one word like anger, uncomfortable, happy, sad, surprised or something else. \n 5. Don't provide whole sentences only the json. \n 6. Make sure that characters dont appear multiple times in the whole JSON. \n 7. follow the rules 1-7 ; \n conversation: {dialog}. ",
        options={"temperature": 0}
    )
    return output['response']

def insert_data():
    postgres = postgres_handler.PostgresHandler()
    mongo = mongo_handler.MongoHandler()

    data = pd.read_csv("the-office_lines.csv")
    chunks = extract_lines_with_two_speaker_changes(data.iloc[0:209])

    for documents in chunks:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=documents.text)
        embedding = response["embedding"]
        postgres_id = postgres.insert_data(embedding, documents.text, documents.episode, documents.season)
        emotions = extract_emotions_from_dialog(documents.text)
        mongo.insert_metadata({"postgres_id": postgres_id, "emotions": emotions})
        print(f"Inserted document with id {postgres_id} and emotions {emotions}")

if __name__ == '__main__':
    postgres_handler = postgres_handler.PostgresHandler()
    mongo_handler = mongo_handler.MongoHandler()
    #insert_data()

