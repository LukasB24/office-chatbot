from pymongo import MongoClient, ASCENDING

class MongoHandler:
    def __init__(self):
        self.client = MongoClient('mongodb://root:example@localhost:27017/')
        self.db = self.client['ragdb']
        self.collection = self.db['emotions']
        self.collection.create_index([("postgres_id", ASCENDING)], unique=True)

    def insert_metadata(self, metadata: dict[str, int or str]):
        try:
            self.collection.insert_one(metadata)
        except Exception as e:
            print(f"Exception while inserting metadata: {e}")

    def get_metadata(self, postgres_id: int):
        try:
            return self.collection.find_one({"postgres_id": postgres_id})
        except Exception as e:
            print(f"Exception while getting metadata: {e}")