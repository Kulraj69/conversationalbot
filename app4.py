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
    api_version="2023-06-01-preview",  # or your API version
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

# Function to generate a response based on the user's question and chat history
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
    chat_box = st.container()  # Container for the chat messages
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

# Show chat history
# display_chat_history()

# CSS to fix the input box at the bottom
st.markdown(
    """
    <style>
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            background-color: white;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the input field as a fixed element
st.markdown('<div class="fixed-input">', unsafe_allow_html=True)

# Use a different session state variable to store the user input temporarily
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  # Initialize if it doesn't exist

# Input field for user input
st.session_state["user_input"] = st.text_area("Type your message here:", height=100, key="user_input_area")

st.markdown('</div>', unsafe_allow_html=True)

# Handle the button click
if st.button("Send"):
    if st.session_state["user_input"]:  # Check if there's input
        response = generate_response(st.session_state["user_input"])
        # Clear the input after sending
        st.session_state["user_input"] = ""  # Clear the stored user input
        # Redisplay chat history to include the new message
        display_chat_history()
