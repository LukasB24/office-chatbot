import postgres_handler
import mongo_handler
import pandas as pd
from file_scraper import chunk_dynamically
import ollama

def extract_emotions_from_dialog(dialog: str):
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
    postgres_handler = postgres_handler.PostgresHandler()
    mongo_handler = mongo_handler.MongoHandler()

    query = "In which season and episode does pam feel uncomfortable?"

    closest_semantic_result = postgres_handler.find_closest_vector(query)
    fetched_emotions = mongo_handler.get_metadata(closest_semantic_result[0])

    query_result = ollama.generate(
        model="llama3.1:8b",
        prompt=f"""
                Using this data: {closest_semantic_result[1]} 
                and this emotion context, if asked for emotions: {fetched_emotions}. 
                This is season: {closest_semantic_result[3]} and this is episode: {closest_semantic_result[4]} use this metadata if asked for. 
                Respond to this prompt: {query}"""
    )

    print(query_result['response'])
    #insert_data()

