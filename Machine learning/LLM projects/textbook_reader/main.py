import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



documents = [uploaded_file.read().decode()]
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts = text_splitter.create_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts,embeddings)
retriever = db.as_retriever
qa = RetrievalQA.from_chain_type(llm = OpenAI(openai_api_key))
qa.run(query_text)

