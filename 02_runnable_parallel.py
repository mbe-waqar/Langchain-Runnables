from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)

prompt2 = PromptTemplate(
    template="What is the currency of {country}?",
    input_variables=["country"]
)

model = ChatOpenAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    RunnableSequence(prompt1, model, parser),
    RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"country": "Pakistan"})
print(result)

print(parallel_chain.get_graph().print_ascii())
# Output: {'What is the capital of Pakistan?': 'Islamabad', 'What is the currency of Pakistan?': 'Pakistani Rupee'}