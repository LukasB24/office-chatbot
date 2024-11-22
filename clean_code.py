import psycopg2
import numpy as np
import ollama
import struct
from sqlalchemy import create_engine
from sqlalchemy.sql import text


# wirklich nur weil wir 45minuten zeit haben anstatt es von Hand in DBeaver zu machen;-)
def postgres_create_db(dbname):
    engine = create_engine("postgresql://user:password@localhost:5432/postgres")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(f"create database {dbname}"))


# das hier ist mehr oder weniger der Code aus Moodle
# Extension reinmachen, Tabelle erstellen, Vektoren einfügen, Vektoren abfragen
def test_postgres_connection():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname='ragdb',
        user='user',
        password='password',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()

    # Enable the pgvector extension (if not already enabled)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create a table with a vector column
    # Achtung: Tabelle hat KEIN document Feld, da hier wirklich nur ein Test der Vector-Funktionalität gemacht wird
    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id SERIAL PRIMARY KEY,
            embedding VECTOR(3)  -- Example: 3-dimensional vector
        );
    """)
    conn.commit()

    # Insert a few sample vectors
    # die Nutzung von numpy ist hier nicht notwendig
    # eine einfache Liste hätte auch gereicht
    sample_vectors = [
        np.array([1.1, 0.2, 0.3], dtype=np.float32),
        np.array([1.4, 0.5, 0.6], dtype=np.float32),
        np.array([1.7, 0.8, 0.9], dtype=np.float32)
    ]

    for vector in sample_vectors:
        cur.execute("INSERT INTO items (embedding) VALUES (%s);", (vector.tolist(),))
    conn.commit()

    # Perform a basic vector similarity query
    query_vector = np.array([0.1, 0.2, 0.25], dtype=np.float32)
    cur.execute("""
        SELECT id, embedding, embedding <-> %s::vector AS distance
        FROM items
        ORDER BY distance
        LIMIT 7;
    """, (query_vector.tolist(),))

    # leider ist das Erebnis "result" kein Dictionary, sondern eine Liste,
    # d.h. wir müssen mit Hilfe von Indizes auf die einzelnen Elemente zugreifen
    # Fetch and display results
    results = cur.fetchall()
    for row in results:
        print(f"ID: {row[0]}, Embedding: {row[1]}, Distance: {row[2]}")

    # Clean up (optional)
    # cur.execute("DROP TABLE IF EXISTS items;")
    # conn.commit()

    # Close connection
    cur.close()
    conn.close()



def test_ollama_embeddings():
    # anstatt mit echten Dokumenten zu arbeiten, wird hier mit einem Dummy-Array gearbeitet
    documents = [
        "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
        "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
        "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
        "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
        "Llamas are vegetarians and have very efficient digestive systems",
        "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
    ]
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname='ragdb',
        user='user',
        password='password',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()
    # diesmal MIT einem document Feld
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
        # print(len(embedding))
        print(embedding)
        cur.execute("INSERT INTO documentvectors2 (embedding, document) VALUES (%s, %s);", (embedding, d))

        # das hier ist der Code aus dem ollama-tutorial https://ollama.com/blog/embedding-models
        # wo sie die chromadb nutzen
        # Chroma sieht wie ein spannendes Projekt aus, ist aber wirklich noch beta, daher erstmal mit
        # "echten" Werkzeugen;-)
        """collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[d]
        )
        """
    conn.commit()
    cur.close()
    conn.close()

# hier wird ein Embedding für eine query erstellt und dann die 3 Dokumente mit den geringsten Distanzen gesucht
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

    # Fetch and display results
    results = cur.fetchall()
    for row in results:
        print(f"ID: {row[0]}, Distance: {row[3]},  Doc: {row[1]}, Embedding: {row[2]}")

    # Clean up (optional)
    # cur.execute("DROP TABLE IF EXISTS items;")
    # conn.commit()

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
    #postgres_create_db('ragdb')
    test_postgres_connection()
    test_ollama_embeddings()
    find_closest_vector("What animals are llamas related to?")
    print(f"Ollama running models: {ollama.ps()}")
