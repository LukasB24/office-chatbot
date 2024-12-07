"""
Datastructures for MonoDB:

{
    "id": "tt1234567",
    "postgres_id": "1", // index needed
    "emotions": {
        "Michael": ["confident", "amused"],
        "Pam": ["uncomfortable"]
    }
}

Datastructures for PostgreSQL:

CREATE TABLE IF NOT EXISTS documentVectors (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1024),  -- Example: 3-dimensional vector
    dialogDocument TEXT,
    episode INTEGER,
    season INTEGER
);
"""
import psycopg2
import ollama

class PostgresHandler:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname='ragdb',
            user='user',
            password='password',
            host='localhost',
            port='5432'
        )
        self.create_table()

    def create_table(self):
        try:
            cursor = self.conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS documentVectors (
                id SERIAL PRIMARY KEY,
                embedding VECTOR(1024),
                dialogDocument TEXT,
                episode INTEGER,
                season INTEGER
            );
            """

            cursor.execute(create_table_query)

            self.conn.commit()
            cursor.close()

        except Exception as e:
            print(f"Exception while creating table: {e}")

    def insert_data(self, embedding, dialog, episode, season) -> int:
        try:
            cursor = self.conn.cursor()

            insert_query = """
                INSERT INTO documentVectors (embedding, dialogDocument, episode, season)
                VALUES (%s, %s, %s, %s)
                RETURNING id;  -- Gibt die ID des eingefügten Dokuments zurück
            """

            cursor.execute(insert_query, (embedding, dialog, episode, season))
            inserted_id = cursor.fetchone()[0]

            self.conn.commit()
            cursor.close()

            return inserted_id
        except Exception as e:
            print(f"Exception while inserting data: {e}")
            return -1

    def find_closest_vector(self, query) -> list:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        embedding = response["embedding"]

        cur = self.conn.cursor()
        cur.execute("""
                  SELECT id, dialogDocument, embedding, embedding <-> %s::vector AS distance
                  FROM documentVectors
                  ORDER BY distance
                  LIMIT 3;
              """, (embedding,))

        results = cur.fetchall()

        self.conn.commit()
        cur.close()

        closest_semantic_result = results[0]
        return closest_semantic_result
