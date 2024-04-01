import streamlit as st
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

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


query = st.text_input("Enter your query")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = get_retriever()
model = OpenAI(temperature=0)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

if st.button("Search"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching..."):
            res = chain.stream(query)
            st.write_stream(res)
