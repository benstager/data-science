from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.prompts.chat import(
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

# SystemMessagePromptTemplate defines what we will be telling openAI
# HumanMessagePromptTemplate is the specific word for the system
# ChatPromptTemplate puts the two in the array together

system_template = 'You are an assistant that returns comma separated lists. You will return a list of length 3.'
system_message = SystemMessagePromptTemplate.from_template(system_template)

human_template = '{text}'
human_message = HumanMessagePromptTemplate.from_template(human_template)

class CSLOutputParser(BaseOutputParser):

    def parse(self, text):
        return text.strip().split(',')
    
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
chain = LLMChain(
    llm = ChatOpenAI(openai_api_key = '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'),
    prompt = chat_prompt,
    output_parser = CSLOutputParser()
)
print(chain.run('fire type pokemon'))
