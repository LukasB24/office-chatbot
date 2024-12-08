import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer

import postgres_handler
import mongo_handler
import ollama

postgres_handler = postgres_handler.PostgresHandler()
mongo_handler = mongo_handler.MongoHandler()

st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("The Office assistant")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

user_input = st.text_input("Enter your question regarding 'The Office'", key="input")
if user_input:
    cypher_query = ""
    database_results = ""

    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            closest_semantic_result = postgres_handler.find_closest_vector(user_input)
            fetched_emotions = mongo_handler.get_metadata(closest_semantic_result[0])

            query_result = ollama.generate(
                model="llama3.1:8b",
                prompt=f"""
                            Using this data: {closest_semantic_result[1]} 
                            and this emotion context, if asked for emotions: {fetched_emotions}. 
                            This is season: {closest_semantic_result[3]} and this is episode: {closest_semantic_result[4]} use this metadata if asked for. 
                            If a query is not specific enough, please ask for more details without telling a emotional context or metadata provided to you.
                            A query that should be more specific could be "How does pam feel about michael" this is a general question and you should ask for a specific situation.
                            Respond to this prompt: {user_input}"""
            )

            st.session_state.system_msgs.append(query_result['response'])
        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key=str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col3:
        if cypher_query:
            st.text_area("Last query", cypher_query, key="_query", height=240)

    with col2:
        if database_results:
            st.text_area("Last answer", database_results, key="_answer", height=240)
