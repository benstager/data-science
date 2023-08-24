import os
import langchain
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TextSplitter
from langchain.chains.summarize import load_summarize_chain

source = input("Please choose a source for documents (home (h), or website (w)): ")
docs = []
files = []
path = ''
os.environ['OPENAI_API_KEY'] = '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'

if source == 'home' or 'h':
    path = input("Please enter a path to search for documents: ")
    print("List of documents in this folder:")
    print(dict(enumerate(os.listdir(path))))
    docs = dict(enumerate(os.listdir(path)))
    num = input("Please select document numbers to analyze, enter 'd' when done: ")
    while num != 'd':
        files.append(docs[int(num)])
        num = input("Please select document numbers to analyze, enter 'd' when done: ")
        
elif source == 'website' or 'w':
    print('STUB')

files_dict = dict(enumerate(files))
documents = []

for file in files:
    splitter = file.split('.')
    if splitter[1] == 'pdf':
        pdf_path = path + '/' + file
        print(pdf_path)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif splitter[1] == '.docx':
        doc_path = path + '/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif splitter[1] == '.txt':
        txt_path = path + '/' + file
        loader = TextLoader(txt_path)
        documents.extend(loader.load())

chain = load_summarize_chain(llm = OpenAI(), chain_type='map_reduce')
summary = chain.run(documents)
print(summary)

print("Your documents have been processed. Summaries are readily available.")

