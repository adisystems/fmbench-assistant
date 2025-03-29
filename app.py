import streamlit as st
import requests
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Georgetown Course Agent",
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
st.title("Georgetown Course Agent")
st.markdown("Ask questions about Georgetown University courses, departments, and programs. You can also ask follow-up questions about courses you've already inquired about.")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False

# API endpoint 
API_URL = "http://localhost:8000/generate"

# Define a helper function for rerunning safely
def safe_rerun():
    try:
        st.experimental_rerun()
    except AttributeError:
        # If experimental_rerun is not available, do nothing.
        pass

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
                     r'<b>\1:</b> \2', 
                     content)
    # Format section headers
    content = re.sub(r'(COURSE DESCRIPTION|PREREQUISITES|COURSE OBJECTIVES|REQUIRED MATERIALS|ADDITIONAL INFORMATION|RESOURCES):', 
                     r'<h4 style="color: #041E42;">\1</h4>', 
                     content)
    # Replace newlines with HTML breaks
    content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return content

def stream_response(question):
    """Make API request to get the chatbot response and simulate streaming."""
    st.session_state.awaiting_response = True
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Create a placeholder for the streaming response
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        payload = {"question": question}
        with st.spinner("Searching for course information..."):
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Expecting response format: {"answer": "..."}
                ai_response = result["answer"]
                words = ai_response.split()
                # Simulate streaming by showing a few words at a time
                for i in range(0, len(words), 3):
                    chunk_size = min(3, len(words) - i)
                    full_response += " ".join(words[i:i+chunk_size]) + " "
                    if i % 9 == 0 or i >= len(words) - 3:
                        formatted_response = format_message(full_response)
                        message_placeholder.markdown(f'<div class="assistant-message">{formatted_response}</div>', unsafe_allow_html=True)
                        time.sleep(0.1)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                message_placeholder.markdown(f'<div class="system-message">Error: {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
    except Exception as e:
        message_placeholder.markdown(f'<div class="system-message">Error: {str(e)}</div>', unsafe_allow_html=True)
    
    st.session_state.awaiting_response = False
    safe_rerun()

def main():
    # Display conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            formatted_content = format_message(message["content"])
            st.markdown(f'<div class="assistant-message">{formatted_content}</div>', unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Input form for user questions
    if not st.session_state.awaiting_response:
        with st.container():
            with st.form(key="query_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Continue the conversation:", 
                    key="user_question", 
                    placeholder="Ask about courses at Georgetown..."
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
            safe_rerun()

# Footer with small print
st.markdown("""
<div class="small-text">
<p>This agent provides information about Georgetown University courses based on publicly available data. 
Information may not be complete or up-to-date. Always verify course details with official Georgetown resources.</p>
</div>
""", unsafe_allow_html=True)

# Run the main app flow
main()