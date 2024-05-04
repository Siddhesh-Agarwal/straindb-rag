import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.v1 import SecretStr


st.title("StrainDB RAG")


@st.cache_resource
def get_retriever():
    """A Chroma retriever that uses OpenAI embeddings"""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        collection_name="strains",
        embedding_function=embeddings,
        persist_directory="./chroma/",
    ).as_retriever()


openai_api_key = st.text_input("Enter your OpenAI API key")
query = st.text_input("Enter your query")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = get_retriever()

if st.button("Search"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key")
        st.stop()
    if not query:
        st.warning("Please enter a query")
        st.stop()
    model = OpenAI(temperature=0, api_key=SecretStr(openai_api_key))
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching..."):
            res = chain.stream(query)
            st.write_stream(res)
