from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI(openai_api_key='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')
chat_model = ChatOpenAI(openai_api_key='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')

llm.predict('name of president')
chat_model.predict('name of president')

from langchain.schema import HumanMessage

text = 'Whats a good name for a Pokemon Title?'
messages = [HumanMessage(content= text)]
llm.predict_messages(messages)

# we can use prompt templates to add in a message to a piece of existent text (like a text corpus)
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template('whats the name of the next {hero} villain?')
prompt.format(hero='batman')

# we can also combine several messages in a list

from langchain.prompts.chat import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

system_template = 'You are {person1} speaking to {person2}'
system_message_template = SystemMessagePromptTemplate.from_template(system_template)

human_template = 'I like {obj}'
human_message_template = HumanMessagePromptTemplate.from_template(human_template)

chat_template = ChatPromptTemplate.from_messages([system_message_template, human_message_template])
chat_template.format_messages(person1 = 'Ben', person2 = 'Hayden', obj = 'Pokemon')

# we can use Output Parsers to into formats that can be used later, using OOP

from langchain.schema import BaseOutputParser

class CSLOutputParser(BaseOutputParser):

    def parse(self, text):
        return text.strip().split(',')

print(CSLOutputParser().parse('My name, is Ben'))

