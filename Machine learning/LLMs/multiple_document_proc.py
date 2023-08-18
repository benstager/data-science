import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import sys

os.environ['OPENAI_API_KEY'] = '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'

documents = []
for file in os.listdir('/Users/benstager/Desktop/docs'):
    if file.endswith('.pdf'):
        pdf_path = '/Users/benstager/Desktop/docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        doc_path = '/Users/benstager/Desktop/docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        txt_path = '/Users/benstager/Desktop/docs/' + file
        loader = TextLoader(txt_path)
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

vector_db = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vector_db.persist()

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=.9, model_name = 'gpt-3.5-turbo'),
    vector_db.as_retriever(search_kwargs={'k':4}),
    return_source_documents = True,
    verbose=False
)

chat_history = []
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')

while True:
    query = input('Please enter your query: ')
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa({'question':query, 'chat_history':chat_history})
    print('Answer:')
    print(result['answer'])
    chat_history.append((query, result['answer']))

