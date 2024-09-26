
import os
from dotenv import load_dotenv  # Import the `dotenv` module
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables from the `.env` file
load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)



llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

st.title("Grocliq GPT")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question):
    llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input)
    st.write(response)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to Punjabi. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg

# print(ai_msg.content)

# response = client.chat.completions.create(
#     model="gpt-35-turbo", # model = "deployment_name".
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
#         {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#         {"role": "user", "content": "Do other Azure AI services support this too?"}
#     ]
# )

# print(response.choices[0].message.content)