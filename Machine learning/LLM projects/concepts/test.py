from langchain.prompts.chat import SystemMessagePromptTemplate, SystemMessage, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [SystemMessage(
        content = ("You are a helpful assistant that re-writes the user's text to"
                "sound more upbeat."
                )

        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

llm = ChatOpenAI(openai_api_key = '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')
print(llm(template.format_messages(text="I dont't like eating yummy things")))