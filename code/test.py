import psycopg2
import numpy as np
# import llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import redis

def postgresVec():
    conn = psycopg2.connect(
        dbname='mydb',
        user='user',
        password='password',
        host='127.0.0.1',
        port='5432'
    )
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id SERIAL PRIMARY KEY,
            embedding VECTOR(3)  -- Example: 3-dimensional vector
        );
    """)
    conn.commit()

    sample_vectors = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
        np.array([0.7, 0.8, 0.9], dtype=np.float32)
    ]

    for vector in sample_vectors:
        cur.execute("INSERT INTO items (embedding) VALUES (%s);", (vector.tolist(),))
    conn.commit()

    query_vector = np.array([0.1, 0.2, 0.25], dtype=np.float32)
    cur.execute("""
        SELECT id, embedding, embedding <-> %s::vector AS distance
        FROM items
        ORDER BY distance
        LIMIT 3;
    """, (query_vector.tolist(),))

    results = cur.fetchall()
    for row in results:
        print(f"ID: {row[0]}, Embedding: {row[1]}, Distance: {row[2]}")

    # Clean up (optional)
    # cur.execute("DROP TABLE IF EXISTS items;")
    # conn.commit()

    # Close connection
    cur.close()
    conn.close()


def llamaletsgo():
    documents = SimpleDirectoryReader("data").load_data()
    Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large", request_timeout=120.0)
    Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)
postgresVec()