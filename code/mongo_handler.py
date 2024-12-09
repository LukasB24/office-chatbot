import json

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

    def get_all_conversations_characters_with_emotions(self) -> list[list[dict[str, str or list[str]]]]:
        pipeline = [
            {
                "$project": {
                    "emotions": 1
                }
            },
            {
                "$unwind": "$emotions"
            },
            {
                "$match": {
                    "$expr": {
                        "$eq": [{"$type": "$emotions.emotions"}, "array"]
                        # Überprüfe, ob `emotions.emotions` ein Array ist
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "conversation": "$_id",
                        "character": "$emotions.character"
                    },
                    "emotions": {
                        "$addToSet": "$emotions.emotions"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.conversation",
                    "characters": {
                        "$push": {
                            "character": "$_id.character",
                            "emotions": {
                                "$reduce": {
                                    "input": "$emotions",
                                    "initialValue": [],
                                    "in": {"$setUnion": ["$$value", "$$this"]}
                                }
                            }
                        }
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "characters": 1
                }
            }
        ]

        result = list(self.collection.aggregate(pipeline))
        return [doc["characters"] for doc in result]

