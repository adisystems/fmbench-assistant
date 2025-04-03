import re
import sys
import time
import datetime
import requests
import argparse
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Georgetown DSAN Assistant",
    page_icon="🐶",  # Bulldog emoji (closest to Hoya mascot)
    layout="centered"
)

# Georgetown colors and custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
        border-color: #041E42;
    }
    .user-message {
        background-color: #C0C0C0;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        color: #041E42;
    }
    .assistant-message {
        background-color: #E0E0E0;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        align-self: flex-end;
        color: #041E42;
        white-space: pre-line;  /* This helps preserve line breaks */
    }
    .timestamp {
        font-size: 10px;
        color: #6A6A6A;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    .user-timestamp {
        text-align: left;
        margin-left: 5px;
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
        color: #041E42; /* Georgetown Blue */
    }
    .system-message {
        color: #6A6A6A;
        font-style: italic;
        text-align: center;
    }
    .stButton button {
        background-color: #041E42;
        color: white;
    }
    .stButton button:hover {
        background-color: #0A3A6D;
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
st.title("Georgetown DSAN Program Agent")
st.markdown("Ask questions about Georgetown University's Data Science and Analytics (DSAN) program. You can also ask follow-up questions about information you've already inquired about.")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = 0

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

# Define a helper function for rerunning safely
def safe_rerun():
    try:
        st.rerun()  # Use the newer st.rerun() instead of experimental_rerun
    except:
        try:
            st.experimental_rerun()  # Fallback for older Streamlit versions
        except:
            pass  # If neither works, just continue

def get_current_timestamp():
    """Get current timestamp in a readable format."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def format_message(message_content):
    """Format message content for better display with HTML."""
    content = message_content
    # Clean up repeated words and section headers
    content = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', content)
    content = re.sub(r'(COURSE DESCRIPTION:?\s*){2,}', r'COURSE DESCRIPTION:', content, flags=re.IGNORECASE)
    content = re.sub(r'(PREREQUISITES:?\s*){2,}', r'PREREQUISITES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(ADDITIONAL INFORMATION:?\s*){2,}', r'ADDITIONAL INFORMATION:', content, flags=re.IGNORECASE)
    content = re.sub(r'(RESOURCES:?\s*){2,}', r'RESOURCES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(COURSE OBJECTIVES:?\s*){2,}', r'COURSE OBJECTIVES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(REQUIRED MATERIALS:?\s*){2,}', r'REQUIRED MATERIALS:', content, flags=re.IGNORECASE)
    
    # Bold certain labels
    content = re.sub(r'(Course|Department|Professor|Schedule|Credits|Location):\s*([^\n]+)', 
                     r'<strong>\1:</strong> \2', 
                     content)
    
    # Format section headers
    content = re.sub(r'(COURSE DESCRIPTION|PREREQUISITES|COURSE OBJECTIVES|REQUIRED MATERIALS|ADDITIONAL INFORMATION|RESOURCES):', 
                     r'<h4 style="color: #041E42; margin-top: 16px; margin-bottom: 8px;">\1</h4>', 
                     content)
    
    # Make double line breaks more visually distinct
    content = content.replace('\n\n', '<br><br>')
    
    # Replace single newlines with HTML breaks
    content = content.replace('\n', '<br>')
    
    return content

def stream_response(question):
    """Make API request to get the chatbot response and simulate streaming."""
    st.session_state.awaiting_response = True
    
    # Add the user's question to the conversation history with timestamp
    timestamp = get_current_timestamp()
    st.session_state.messages.append({"role": "user", "content": question, "timestamp": timestamp})
    
    # Create a placeholder for the streaming response
    message_placeholder = st.empty()
    
    try:
        # Updated payload to include thread_id for conversation memory
        payload = {
            "question": question,
            "thread_id": st.session_state.thread_id
        }
        
        with st.spinner("Searching for DSAN program information..."):
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Updated to handle the new response format
                outputs = result.get("result", [])
                
                # Find the AI's response (it will be the last 'ai' message)
                ai_messages = [msg for msg in outputs if msg["role"] == "ai"]
                if ai_messages:
                    ai_response = ai_messages[-1]["content"]
                    
                    # Apply formatting to the complete response once
                    formatted_full_response = format_message(ai_response)
                    
                    # Get timestamp for the assistant's response
                    response_timestamp = get_current_timestamp()
                    
                    # Store original with timestamp for session
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_response,
                        "timestamp": response_timestamp
                    })
                    
                    # Display the entire formatted response at once
                    message_placeholder.markdown(
                        f'<div class="assistant-message">{formatted_full_response}</div>' + 
                        f'<div class="timestamp assistant-timestamp">{response_timestamp}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    message_placeholder.markdown(
                        f'<div class="system-message">No response from the assistant.</div>', 
                        unsafe_allow_html=True
                    )
            else:
                message_placeholder.markdown(
                    f'<div class="system-message">Error: {response.status_code} - {response.text}</div>', 
                    unsafe_allow_html=True
                )
    except Exception as e:
        message_placeholder.markdown(
            f'<div class="system-message">Error: {str(e)}</div>', 
            unsafe_allow_html=True
        )
    
    st.session_state.awaiting_response = False
    safe_rerun()

def main():
    # Display conversation history
    for message in st.session_state.messages:
        timestamp = message.get("timestamp", "")
        
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">{message["content"]}</div>' + 
                f'<div class="timestamp user-timestamp">{timestamp}</div>', 
                unsafe_allow_html=True
            )
        else:
            formatted_content = format_message(message["content"])
            st.markdown(
                f'<div class="assistant-message">{formatted_content}</div>' + 
                f'<div class="timestamp assistant-timestamp">{timestamp}</div>', 
                unsafe_allow_html=True
            )
    
    # Add some spacing
    st.write("")
    st.write("")
    
    # Input form for user questions
    if not st.session_state.awaiting_response:
        with st.container():
            with st.form(key="query_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Continue the conversation:", 
                    key="user_question", 
                    placeholder="Ask about the DSAN program at Georgetown..."
                )
                col1, col2 = st.columns([4, 1])
                with col2:
                    submit_button = st.form_submit_button("Submit")
                if submit_button and user_input:
                    st.session_state.pending_question = user_input
                    safe_rerun()
    
    # Process pending question if present
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        stream_response(question)
    
    # Button to reset the conversation
    if not st.session_state.awaiting_response:
        if st.button("Start New Conversation"):
            st.session_state.messages = []
            # Generate a new thread ID when starting a new conversation
            st.session_state.thread_id = st.session_state.get('thread_id', 0) + 1
            safe_rerun()

# Footer with small print
st.markdown("""
<div class="small-text">
<p>This agent provides information about Georgetown University's DSAN program based on publicly available data. 
Information may not be complete or up-to-date. Always verify details with official Georgetown resources.</p>
</div>
""", unsafe_allow_html=True)

# Run the main app flow
if __name__ == "__main__":
    main()