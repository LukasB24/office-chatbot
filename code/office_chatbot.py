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

user_input = st.text_input("Enter your question regarding 'The Office'", key="input")

if user_input:
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
                            A query that should be more specific could be "How does pam feel about michael?". This is a general question and you should ask if it's possible to specify the question and ask again.
                            A query like "In which season and episode does pam feel uncomfortable?" is specific enough tough and you should answer with the episode and season.
                            A more specific question could be "How does pam feel about michael when he introduces her and why?" This question is more specific and can be answered more accurately.
                            It is important that you only answer the question that was asked and not provide additional information like emotion context or metadata that was not asked for. 
                            If the question is not specific enough only ask for clarification without giving an answer to the question.
                            Answer short and precise.
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
            for i in range(len(st.session_state["system_msgs"])):
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["system_msgs"][i], key=str(i) + "_assistant")
