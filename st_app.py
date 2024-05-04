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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return Chroma(
        collection_name="strains",
        embedding_function=embeddings,
        persist_directory="./chroma/",
    ).as_retriever()


openai_api_key = st.text_input(
    label="Enter your OpenAI API key",
    max_chars=51,
    help="Get an API key from https://platform.openai.com/account/api-keys",
    type="password",
    placeholder="sk-...",
)
query = st.text_input(
    label="Enter your query",
    help="Search for strains",
    placeholder="What are the medical uses of Stunna",
)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

if st.button("Search"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key")
        st.stop()
    if not query:
        st.warning("Please enter a query")
        st.stop()

    api_key = SecretStr(openai_api_key)
    retriever = get_retriever()
    model = OpenAI(temperature=0, api_key=api_key)
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
            st.write_stream(chain.stream(query))
