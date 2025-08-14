from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(
    prompt, model, parser
)

result = chain.invoke({"country": "Pakistan"})

print(result)