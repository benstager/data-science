from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from unstructured.cleaners.core import remove_punctuation,clean,clean_extra_whitespace
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os

api_key = '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'

def generate_document(url):
    loader = UnstructuredURLLoader(urls=[url],
                mode="elements",
                post_processors=[clean,remove_punctuation,clean_extra_whitespace])
    elements = loader.load()
    selected_elements = [e for e in elements if e.metadata['category']=="NarrativeText"]
    full_clean = " ".join([e.page_content for e in selected_elements])
    return Document(page_content=full_clean, metadata={"source":url})


def summarize_document(url):
    "Given an URL return the summary from OpenAI model"
    llm = OpenAI(model_name='ada',temperature=0,openai_api_key=api_key)
    chain = load_summarize_chain(llm, chain_type="stuff")
    tmp_doc = generate_document(url)
    summary = chain.run([tmp_doc])
    return clean_extra_whitespace(summary)

summarize_document('https://www.annualreviews.org/doi/abs/10.1146/annurev-polisci-102512-194818')