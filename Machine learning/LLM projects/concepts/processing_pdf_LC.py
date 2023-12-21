"""
Attempt to read in a pdf and extract pertinent information from it using LLM
"""
import os
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = "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"

pdf_loader = PyPDFLoader("/Users/benstager/Desktop/sample_text_corpus.pdf")
documents = pdf_loader.load()

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm = OpenAI())
query = "What is the professor's name? If it's not mentioned say N/A"
response = chain.run(input_documents=documents,question=query)

# Now lets try to implement embeddings and vector databases
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# chunks of 1000 characters, with overlap of 200 characters
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# store in vector database
vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="/Users/benstager/Desktop/Python\ Code/Data\ Science/Machine\ learning/LLMs"
)

vectordb.persist()

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    retriever = vectordb.as_retriever(search_kwargs={'k':5}),
    return_source_documents = True
)

# we can now run queries on our document
result = qa_chain({'query':'When is the class scheduled?'})

# suppose we want to use .pdf, .docx, and .txt, we can iterate over them based on their ending clause
