from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.agents import load_tools


#from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="EleutherAI/gpt-j-6B", model_kwargs={"temperature":0.7, "max_length":512}, huggingfacehub_api_token="your_api_token_here")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

tools = load_tools(["serpapi", "llm-math"], llm = llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

result = agent.invoke("What is the capital of France?")
print('----------------------------------')
print(result)
print('**********************************')

chain = SequentialChain(
    chains=[
        ("question", prompt),
        ("result", agent)
    ],
    input_variables=["question"],
    output_variables=["result"],
    verbose=True
)

result = chain({"question": "What is the capital of France and what is the population?"})
print(result["result"])
