# conda create -n new_env python=3.10
# conda activate new_env
# export USER_AGENT="AnyPageChatBot/0.0.1 (Mac; Python 3.10)"
# streamlit run src/app.py
# ~/.streamlit/
# pip install streamlit langchain langchain-core langchain-community langchain-openai openai beautifulsoup4 python-dotenv chromadb streamlit googleapis-common-protos protobuf streamlit-javascript pyyaml ua-parser user-agents
# pip freeze | grep streamlit >> requirements.txt
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from datetime import datetime

load_dotenv()

def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()

        if not document:
            return None

        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)

        if not document_chunks:
            return None

        vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
        return vector_store
    except Exception as e:
        st.error(f"Error processing URL {url}: {str(e)}")
        return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini"

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# app config
st.set_page_config(page_title="Any page bot", page_icon="🤖")
st.title("Any page bot 🤖 ")

website_url = st.text_input("Enter web address you want to chat with", placeholder="https://", label_visibility="collapsed")

if website_url is None or website_url == "":
    st.info("Enter web address you want to chat with")
else:
    if "prev_url" not in st.session_state or st.session_state.prev_url != website_url:
        st.session_state.prev_url = website_url
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am Linko. What do you want to know about this webpage?"),
        ]
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

        if st.session_state.vector_store is None:
            st.error("Failed to process the webpage. Please try a different URL.")
        else:
            st.success("Webpage processed successfully!")

    if st.session_state.get("vector_store") is not None:
        # user input
        user_query = st.chat_input("Explain me this content briefly")
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