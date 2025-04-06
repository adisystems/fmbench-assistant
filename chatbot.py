import re
import sys
import time
import datetime
import requests
import argparse
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="FMBench Assistant",
    page_icon="📊",  # Chart emoji representing benchmarking
    layout="centered"
)

# Custom CSS for styling (maintaining the chat UI)
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
        border-color: #232F3E;  # AWS color
    }
    .timestamp {
        font-size: 10px;
        color: #6A6A6A;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    .user-timestamp {
        text-align: right;
        margin-right: 5px;
    }
    .assistant-timestamp {
        text-align: right;
        margin-right: 5px;
    }
    .small-text {
        font-size: 12px;
        color: #6A6A6A;
    }
    h1 {
        color: #232F3E; /* AWS Blue */
    }
    .system-message {
        color: #6A6A6A;
        font-style: italic;
        text-align: center;
    }
    .stButton button {
        background-color: #232F3E;
        color: white;
    }
    .stButton button:hover {
        background-color: #31465F;
        color: white;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 20px;
        background-color: white;
        border-top: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("FMBench Assistant")
st.markdown("Ask questions about the Foundation Model Benchmarking Tool (FMBench). You can ask about its features, usage, and general capabilities while respecting security and implementation boundaries.")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = 0
if 'display_messages' not in st.session_state:
    st.session_state.display_messages = []

# Get command line arguments safely
def get_args():
    parser = argparse.ArgumentParser(description="Streamlit app with command line arguments")
    parser.add_argument("--api-server-url", type=str, default='http://localhost:8000/generate', 
                       help="API server URL")
    
    # Handle argument parsing in a way that works with Streamlit
    try:
        args_list = []
        # Look for arguments after the script name or after "--"
        if "--" in sys.argv:
            args_idx = sys.argv.index("--") + 1
            args_list = sys.argv[args_idx:]
        elif len(sys.argv) > 1:
            # Try to extract args that look like they're meant for our script
            for i, arg in enumerate(sys.argv[1:], 1):
                if arg.startswith("--api-server-url"):
                    if "=" in arg:
                        args_list.append(arg)
                    elif i < len(sys.argv) - 1:
                        args_list.extend([arg, sys.argv[i+1]])
        
        return parser.parse_args(args_list)
    except Exception as e:
        # Fallback to default if any issues with arg parsing
        st.warning(f"Argument parsing issue: {str(e)}. Using default API URL.")
        return parser.parse_args([])

# Get the arguments
args = get_args()
API_URL = args.api_server_url

def get_current_timestamp():
    """Get current timestamp in a readable format."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def process_response(question):
    """Make API request to get the chatbot response."""
    # Add the user's question to the conversation history with timestamp
    timestamp = get_current_timestamp()
    
    # First display the user message immediately
    with st.chat_message("user"):
        st.write(question)
        st.caption(f"{timestamp}")
    
    # Add to session state
    st.session_state.messages.append({"role": "user", "content": question, "timestamp": timestamp})
    
    # Create placeholder for assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Updated payload to include thread_id for conversation memory
            payload = {
                "question": question,
                "thread_id": st.session_state.thread_id
            }
            
            with st.spinner("Searching for FMBench information..."):
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    # Updated to handle the new response format
                    outputs = result.get("result", [])
                    
                    # Find the AI's response (it will be the last 'ai' message)
                    ai_messages = [msg for msg in outputs if msg["role"] == "ai"]
                    if ai_messages:
                        ai_response = ai_messages[-1]["content"]
                        
                        # Get timestamp for the assistant's response
                        response_timestamp = get_current_timestamp()
                        
                        # Update the placeholder with the actual response
                        message_placeholder.markdown(ai_response)
                        st.caption(f"{response_timestamp}")
                        
                        # Store original with timestamp for session
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ai_response,
                            "timestamp": response_timestamp
                        })
                    else:
                        message_placeholder.error("No response from the assistant.")
                else:
                    message_placeholder.error(f"Error: {response.status_code} - {response.text}")
            
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.messages:
        timestamp = message.get("timestamp", "")
        
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                st.caption(f"{timestamp}")
        else:
            with st.chat_message("assistant"):
                # Use Streamlit's native markdown rendering
                st.markdown(message["content"])
                st.caption(f"{timestamp}")

def main():
    # Display existing chat history first
    if len(st.session_state.messages) > 0 and not st.session_state.awaiting_response:
        display_chat_history()
    
    # Process pending question if present
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        st.session_state.awaiting_response = True
        
        # Process the response (displays both user question and assistant answer)
        process_response(question)
        
        # Mark that we're done processing
        st.session_state.awaiting_response = False
    
    # Input for user questions
    if not st.session_state.awaiting_response:
        # Use Streamlit's chat_input for a more natural chat interface
        user_input = st.chat_input("Ask about FMBench features and capabilities...")
        if user_input:
            st.session_state.pending_question = user_input
            st.rerun()
    
    # Button to reset the conversation
    if not st.session_state.awaiting_response:
        if st.sidebar.button("Start New Conversation"):
            st.session_state.messages = []
            # Generate a new thread ID when starting a new conversation
            st.session_state.thread_id = st.session_state.get('thread_id', 0) + 1
            st.rerun()

# Footer with small print
st.sidebar.markdown("""
### About
This agent provides information about Foundation Model Benchmarking Tool (FMBench) based on publicly available documentation.

For the most up-to-date information and specific implementation details, please refer to the [official documentation](https://aws-samples.github.io/foundation-model-benchmarking-tool/).
""")

# Run the main app flow
if __name__ == "__main__":
    main()