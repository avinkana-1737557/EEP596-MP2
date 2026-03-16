import os
import streamlit as st
from head_agent import Head_Agent

st.set_page_config(page_title="ML Chatbot", page_icon="🤖")
st.title("Mini Project 2: Multi-Agent Chatbot")


def get_secret(name: str, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


OPENAI_KEY = get_secret("OPENAI_API_KEY")
PINECONE_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX = get_secret("PINECONE_INDEX", "machine-learning-doc")
PINECONE_NAMESPACE = get_secret("PINECONE_NAMESPACE", "ns1000")

if not OPENAI_KEY or not PINECONE_KEY:
    st.error(
        "Missing required secrets. Add OPENAI_API_KEY and PINECONE_API_KEY in Streamlit Cloud app settings."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.sidebar.button("Clear conversation"):
    st.session_state["messages"] = []
    if "head_agent" in st.session_state:
        del st.session_state["head_agent"]
    st.rerun()

if "head_agent" not in st.session_state:
    head = Head_Agent(
        openai_key=OPENAI_KEY,
        pinecone_key=PINECONE_KEY,
        pinecone_index_name=PINECONE_INDEX,
        pinecone_namespace=PINECONE_NAMESPACE,
    )
    head.setup_sub_agents()
    st.session_state["head_agent"] = head

head_agent = st.session_state["head_agent"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    try:
        response = head_agent.main_loop(
            user_query=user_input,
            conversation_history=[
                m["content"]
                for m in st.session_state["messages"]
                if m["role"] == "user"
            ][:-1],
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
