# conda create -n new_env python=3.10
# conda activate new_env
# export USER_AGENT="AnyPageChatBot/0.0.1 (Mac; Python 3.10)"
# streamlit run src/app.py
# ~/.streamlit/
# pip install streamlit langchain langchain-core langchain-community langchain-openai openai beautifulsoup4 python-dotenv chromadb streamlit googleapis-common-protos protobuf
# pip freeze | grep streamlit >> requirements.txt

import streamlit as st
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

    return response['answer']

# JavaScript для анімації сайдбару
sidebar_animation = """
<script>
function animateSidebar() {
    const sidebar = window.parent.document.querySelector('.css-1544g2n.e1fqkh3o4');
    if (sidebar) {
        setTimeout(() => {
            sidebar.style.transition = 'transform 0.3s ease-in-out';
            sidebar.style.transform = 'translateX(20px)';
            setTimeout(() => {
                sidebar.style.transform = 'translateX(0px)';
                setTimeout(() => {
                    sidebar.style.transform = 'translateX(20px)';
                    setTimeout(() => {
                        sidebar.style.transform = 'translateX(0px)';
                    }, 300);
                }, 300);
            }, 300);
        }, 2000);
    }
}
animateSidebar();
</script>
"""

# app config
st.set_page_config(page_title="Any Page Chatbot", page_icon="🤖")
st.title("Any Page Chatbot")

# Перевірка на мобільний пристрій і вставка JavaScript
if st.is_mobile():
    st.components.v1.html(sidebar_animation, height=0)

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Web page URL")

if website_url is None or website_url == "":
    st.info("Please enter a web page URL")
    if st.is_mobile():
        st.text("by clicking on > in the corner ↖")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
