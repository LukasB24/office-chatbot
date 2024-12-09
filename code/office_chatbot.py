from timeit import default_timer as timer

import ollama
import streamlit as st
from streamlit_chat import message

import mongo_handler
import postgres_handler
import redis_handler

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama

neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"

llm = ChatOllama(model="llama3.1:8b")

cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher queries compatible ONLY for Neo4j Version 5.
2. Do not use EXISTS, SIZE, or HAVING keywords in the Cypher. Use aliases when using the WITH keyword.
3. Use only Nodes and relationships explicitly mentioned in the schema.
4. Always perform case-insensitive and fuzzy searches for properties using `toLower` and `contains`. 
   For example:
   - To search for a Client, use `toLower(client.id) contains 'neo4j'`.
   - To search for Slack Messages, use `toLower(SlackMessage.text) contains 'neo4j'`.
   - To search for a project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.
5. Never use relationships that are not explicitly mentioned in the schema.
6. Use `COUNT` and aggregation with `ORDER BY` and `LIMIT` when summarizing data or identifying the top result.
7. Always use descriptive aliases for returned fields.

schema: {schema}

Examples:
Question: Which client's projects use most of our people?
Answer: ```MATCH (c:CLIENT)<-[:HAS_CLIENT]-(p:Project)-[:HAS_PEOPLE]->(person:Person)
RETURN c.name AS Client, COUNT(DISTINCT person) AS NumberOfPeople
ORDER BY NumberOfPeople DESC```
Question: Which person uses the largest number of different technologies?
Answer: ```MATCH (person:Person)-[:USES_TECH]->(tech:Technology)
RETURN person.name AS PersonName, COUNT(DISTINCT tech) AS NumberOfTechnologies
ORDER BY NumberOfTechnologies DESC```
Question: How do characters feel when Michael is part of a conversation?
Answer:
```MATCH (michael:Character)
WHERE toLower(michael.name) = toLower("Michael")
MATCH (michael)-[:PARTICIPATES_IN]->(conversation:Conversation)<-[:PARTICIPATES_IN]-(other:Character)
MATCH (other)-[feels:FEELS]->(conversation)
RETURN other.name AS character, feels.emotion AS emotion
ORDER BY character```
Question: Which character participates in the most conversations? 
Answer:
```MATCH (character:Character)-[:PARTICIPATES_IN]->(conversation:Conversation)
RETURN character.name AS character, COUNT(conversation) AS numberOfConversations
ORDER BY numberOfConversations DESC
LIMIT 1```

Question: {question}
"""

cypher_prompt = PromptTemplate(
    template = cypher_generation_template,
    input_variables = ["schema", "question"]
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

def query_graph(user_query):
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True
        )
    result = chain(user_query)
    return result

postgres_handler = postgres_handler.PostgresHandler()
mongo_handler = mongo_handler.MongoHandler()
redis_handler = redis_handler.RedisHandler()

st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("The Office assistant")

user_input = st.text_input("Enter your question regarding 'The Office'", key="input")

if user_input:
    cypher_query = ""
    database_results = ""
    final_answer = ""

    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        cached_response = redis_handler.get_data(user_input)

        if cached_response:
            st.session_state.system_msgs.append(cached_response)
        else:
            try:
                result = query_graph(user_input)

                intermediate_steps = result["intermediate_steps"]
                cypher_query = intermediate_steps[0]["query"]
                database_results = intermediate_steps[1]["context"]

                final_answer = result["result"]

                if "don't know" in final_answer:
                    raise Exception("Answer not found")

            except Exception:
                print("Answer not found in graph, try to find result using embeddings")
                response = ollama.embeddings(model="mxbai-embed-large", prompt=user_input)
                embedding = response["embedding"]

                closest_semantic_result = postgres_handler.find_closest_vector(embedding)
                fetched_emotions = mongo_handler.get_metadata(closest_semantic_result[0])

                query_result = ollama.generate(
                    model="llama3.1:8b",
                    prompt=f"""
                                Using this data: {closest_semantic_result[1]} 
                                and this emotion context, if asked for emotions: {fetched_emotions}. 
                                This is season: {closest_semantic_result[3]} and this is episode: {closest_semantic_result[4]} use this metadata if asked for. 
                                If a query is not specific enough, please ask for more details without telling a emotional context or metadata provided to you.
                                A query that should be more specific could be "How does pam feel about michael?". This is a general question and you should ask if it's possible to specify the question and ask again.
                                A query like "In which season and episode does pam feel uncomfortable?" is specific enough tough and you should answer with the episode and season.
                                A more specific question could be "How does pam feel about michael when he introduces her and why?" This question is more specific and can be answered more accurately.
                                It is important that you only answer the question that was asked and not provide additional information like emotion context or metadata that was not asked for. 
                                If the question is not specific enough only ask for clarification without giving an answer to the question.
                                Answer short and precise.
                                Respond to this prompt: {user_input}"""
                )

                final_answer = query_result['response']

            finally:
                redis_handler.set_data(user_input, final_answer)
                st.session_state.system_msgs.append(final_answer)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"])):
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["system_msgs"][i], key=str(i) + "_assistant")
