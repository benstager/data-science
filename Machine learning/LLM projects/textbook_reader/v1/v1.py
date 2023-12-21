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
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import MathpixPDFLoader

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

# override
files_dict = dict(enumerate(files))
documents = []

#MathPixPDFLoader
for file in files:
    splitter = file.split('.')
    if splitter[1] == 'pdf':
        pdf_path = path + '/' + file
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
print(len(documents))

query = input("Ask Agent E for a random question or specific topic question: ")
chain = load_summarize_chain(llm = OpenAI(), chain_type='map_reduce')
chain1 = load_qa_chain(llm = OpenAI(), chain_type='stuff')
#summary = chain.run(documents)
qa = chain1.run(input_documents = documents,question=query)

with open('calculus.txt', 'w') as f:
    f.write(qa)
