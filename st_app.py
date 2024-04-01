import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("StrainDB RAG")


@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        collection_name="strains",
        embedding_function=embeddings,
        persist_directory="./chroma/",
    )


query = st.text_input("Enter your query")

if st.button("Search"):
    if not query:
        st.warning("Please enter a query")
    else:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)
        st.json([doc.json() for doc in docs])
