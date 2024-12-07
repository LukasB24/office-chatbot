import numpy as np
import ollama
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.sql import text


def postgres_create_db(dbname):
    engine = create_engine("postgresql://user:password@localhost:5432/postgres")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(f"create database {dbname}"))

def test_ollama_embeddings():
    documents = [
        "Michael: All right Jim. Your quarterlies look very good. How are things at the library? Jim: Oh, I told you. I couldn’t close it. So… Michael: So you’ve come to the master for guidance? Is this what you’re saying, grasshopper? Jim: Actually, you called me in here, but yeah.",
        "Michael: All right. Well, let me show you how it’s done. Michael:  [on the phone] Yes, I’d like to speak to your office manager, please. Yes, hello. This is Michael Scott. I am the Regional Manager of Dunder Mifflin Paper Products. Just wanted to talk to you manager-a-manger. [quick cut scene] All right. Done deal. Thank you very much, sir. You’re a gentleman and a scholar. Oh, I’m sorry. OK. I’m sorry. My mistake. [hangs up] That was a woman I was talking to, so… She had a very low voice. Probably a smoker, so… [Clears throat] So that’s the way it’s done. Michael:  I’ve, uh, I’ve been at Dunder Mifflin for 12 years, the last four as Regional Manager. If you want to come through here… See we have the entire floor. So this is my kingdom, as far as the eye can see. This is our receptionist, Pam. Pam! Pam-Pam! Pam Beesly. Pam has been with us for…  forever. Right, Pam? Pam: Well. I don’t know. Michael: If you think she’s cute now, you should have seen her a couple of years ago. [growls] Pam: What?",
        "Michael: Any messages? Pam: Uh, yeah. Just a fax. Michael: Oh, Pam. This is from corporate. How many times have I told you? There’s a special filing cabinet for things from corporate. Pam: You haven’t told me."
    ]

    conn = psycopg2.connect(
        dbname='ragdb',
        user='user',
        password='password',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()
    cur.execute("""
            CREATE TABLE IF NOT EXISTS documentvectors2 (
                id SERIAL PRIMARY KEY,
                embedding VECTOR(1024),  -- Example: 3-dimensional vector
                document TEXT
            );
        """)
    conn.commit()

    for i, d in enumerate(documents):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
        embedding = response["embedding"]
        cur.execute("INSERT INTO documentvectors2 (embedding, document) VALUES (%s, %s);", (embedding, d))

    conn.commit()
    cur.close()
    conn.close()

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

def find_closest_vector(query):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    embedding = response["embedding"]

    conn = psycopg2.connect(
        dbname='ragdb',
        user='user',
        password='password',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()
    cur.execute("""
            SELECT id, document, embedding, embedding <-> %s::vector AS distance
            FROM documentvectors2
            ORDER BY distance
            LIMIT 3;
        """, (embedding,))

    results = cur.fetchall()
    for row in results:
        print(f"ID: {row[0]}, Distance: {row[3]},  Doc: {row[1]}, Embedding: {row[2]}")

    cur.execute("DROP TABLE IF EXISTS documentvectors2;")
    conn.commit()

    closest_semantic_result = results[0]
    emotions = extract_emotions_from_dialog(closest_semantic_result[1])

    print(closest_semantic_result[1])
    print(emotions)

    cur.close()
    conn.close()

    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"Using this data: {closest_semantic_result[1]} and this emotion context: {emotions}. Respond to this prompt: {query}"
    )

    print(output['response'])

if __name__ == '__main__':
    #postgres_create_db('ragdb')
    test_ollama_embeddings()
    # find_closest_vector("Who is the chief of the company?")
    find_closest_vector("How does pam feel about michael in this dialog?")

