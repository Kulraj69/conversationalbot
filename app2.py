import os
from dotenv import load_dotenv  # Import the `dotenv` module
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables from the `.env` file
load_dotenv()

# Initialize the AzureOpenAI client
client = AzureOpenAI(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)

# Configure the AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Create a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Initialize the chat history
if "history" not in st.session_state:
    st.session_state["history"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Function to generate response based on the user's question and chat history
def generate_response(question):
    # Add user message to the history
    st.session_state["history"].append({"role": "user", "content": question})
    
    # Pass the history to the model
    response = llm.invoke(st.session_state["history"])
    
    # Add the assistant's response to the history
    st.session_state["history"].append({"role": "assistant", "content": response.content})
    
    return response.content

# Streamlit UI
st.title("Grocliq GPT with Memory")

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input)
    st.write(response)

# Option to show chat history
if st.checkbox("Show Chat History"):
    for message in st.session_state["history"]:
        st.write(f"{message['role'].capitalize()}: {message['content']}")
