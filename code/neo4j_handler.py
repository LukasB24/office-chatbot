from py2neo import Graph, Node, Relationship


class Neo4jHandler:
    def __init__(self):
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

    def insert_conversation_characters_with_feelings(self, characters_with_feelings: list):
        for i, characters in enumerate(characters_with_feelings):
            conversation_node = Node("Conversation", name=f"Conversation {i+1}")
            self.graph.create(conversation_node)

            for char in characters:
                character_node = Node("Character", name=char['character'])
                self.graph.create(character_node)

                participates_in_rel = Relationship(character_node, "PARTICIPATES_IN", conversation_node)
                self.graph.create(participates_in_rel)

                for emotion in char['emotions']:
                    feels_rel = Relationship(character_node, "FEELS", conversation_node)
                    feels_rel["emotion"] = emotion
                    self.graph.create(feels_rel)


