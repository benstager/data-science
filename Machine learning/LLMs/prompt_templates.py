from langchain import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a {typ} Pokemon")
prompt_template.format(typ='Legendary')

# can leave argument empty in .format

prompt_template.format()

from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_template(
    ("system", "You are a supervillain"),
    ("human", "Hello how are you?"),
    ("ai", "I am an evil person")
)
