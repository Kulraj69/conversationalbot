import os
from dotenv import load_dotenv  # Import the `dotenv` module
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
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
    max_tokens=150,  # Set a max token limit to avoid issues
    timeout=None,
    max_retries=2,
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
st.set_page_config(page_title="Grocliq GPT with Memory", page_icon="🤖", layout="wide")

st.title("Grocliq GPT with Memory")

# Display chat history in a scrollable container
def display_chat_history():
    # Create a scrollable area for chat messages
    chat_area = st.container()
    with chat_area:
        chat_box = st.empty()  # Placeholder for chat messages
        chat_messages = []  # List to hold chat messages

        for message in st.session_state["history"]:
            if message["role"] == "user":
                chat_messages.append(
                    f'<div style="text-align: right; padding: 10px; background-color: black; border-radius: 10px; margin: 5px;">**You:** {message["content"]}</div>'
                )
            else:
                chat_messages.append(
                    f'<div style="text-align: left; padding: 10px; background-color: green; border-radius: 10px; margin: 5px;">**Assistant:** {message["content"]}</div>'
                )

        # Show chat messages in a scrollable box
        chat_box.markdown(
            f'<div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">{" ".join(chat_messages)}</div>',
            unsafe_allow_html=True
        )

# Show chat history by default
# if st.checkbox("Show Chat History", value=True):
#     display_chat_history()

# Input area for user to type their query
user_input = st.text_area("Type your message here:", height=100)

# Handle the button click
if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        # Clear the input after sending
        st.session_state.user_input = ""  # Clear the stored user input
        # Redisplay chat history to include the new message
        display_chat_history()

# Optionally, if you want to display the response directly after sending
# if user_input:
#     st.write(response)
