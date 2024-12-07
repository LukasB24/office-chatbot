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
        "Michael: All right Jim. Your quarterlies look very good. How are things at the library? Jim: Oh, I told you. I couldn’t close it. So…",
        "Michael: So you’ve come to the master for guidance? Is this what you’re saying, grasshopper? Jim: Actually, you called me in here, but yeah.",
        "Michael: All right. Well, let me show you how it’s done. Michael:  [on the phone] Yes, I’d like to speak to your office manager, please. Yes, hello. This is Michael Scott. I am the Regional Manager of Dunder Mifflin Paper Products. Just wanted to talk to you manager-a-manger. [quick cut scene] All right. Done deal. Thank you very much, sir. You’re a gentleman and a scholar. Oh, I’m sorry. OK. I’m sorry. My mistake. [hangs up] That was a woman I was talking to, so… She had a very low voice. Probably a smoker, so… [Clears throat] So that’s the way it’s done. Michael:  I’ve, uh, I’ve been at Dunder Mifflin for 12 years, the last four as Regional Manager. If you want to come through here… See we have the entire floor. So this is my kingdom, as far as the eye can see. This is our receptionist, Pam. Pam! Pam-Pam! Pam Beesly. Pam has been with us for…  forever. Right, Pam? Pam: Well. I don’t know.",
        "Michael: If you think she’s cute now, you should have seen her a couple of years ago. [growls] Pam: What?",
        "Michael: Any messages? Pam: Uh, yeah. Just a fax.",
        "Michael: Oh, Pam. This is from corporate. How many times have I told you? There’s a special filing cabinet for things from corporate. Pam: You haven’t told me."
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
        print(embedding)
        cur.execute("INSERT INTO documentvectors2 (embedding, document) VALUES (%s, %s);", (embedding, d))

    conn.commit()
    cur.close()
    conn.close()

def find_closest_vector(query):
    print(f"embedding {query} = ", end="")
    response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    embedding = response["embedding"]
    print(f" {embedding}")
    print(f"Ollama running models: {ollama.ps()}")

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

    # Clean up (optional)
    cur.execute("DROP TABLE IF EXISTS documentvectors2;")
    conn.commit()

    # Close connection
    cur.close()
    conn.close()
    # generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"Using this data: {row[1]}. Respond to this prompt: {query}"
    )

    print(output['response'])

    # response without rag - nur zum Vergleich wie der Output mit und ohne RAG aussieht
    print("and this is the response without RAG")
    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"Respond to this prompt: {query}"
    )

    print(output['response'])

if __name__ == '__main__':
    # postgres_create_db('ragdb')
    test_ollama_embeddings()
    # find_closest_vector("Who is the chief of the company?")
    find_closest_vector("How does pam feel about michael?")
    # print(f"Ollama running models: {ollama.ps()}")

    person = "Pam"
    statement = """Michael:  I’ve, uh, I’ve been at Dunder Mifflin for 12 years, the last four as Regional Manager. If you want to come through here… See we have the entire floor. So this is my kingdom, as far as the eye can see. This is our receptionist, Pam. Pam! Pam-Pam! Pam Beesly. Pam has been with us for…  forever. Right, Pam?
                 Pam: Well. I don’t know."
                 Michael: If you think she’s cute now, you should have seen her a couple of years ago. [growls] 
                 Pam: What?"""
    output = ollama.generate(
        model="llama3.1:8b",
        prompt=f"What is the emotion of {person} in this conversation, describe with one word like anger, uncomfortable, happy, sad, surprised or something else: {statement}"
    )
    print(output['response'])
