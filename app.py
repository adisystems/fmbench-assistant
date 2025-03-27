import streamlit as st
import requests
import json
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Georgetown Course Agent",
    page_icon="🦅",
    layout="centered"
)

# Georgetown colors
# Primary: #041E42 (Georgetown Blue)
# Secondary: #6A6A6A (Gray)
# Accent: #C0C0C0 (Light Gray)

# Custom CSS for Georgetown-themed appearance
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
        color: #041E42; /* Georgetown blue */
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
</style>
""", unsafe_allow_html=True)

# App title
st.title("Georgetown Course Agent")
st.markdown("Ask questions about Georgetown University courses, departments, and programs.")

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = 0

# API endpoint 
API_URL = "http://localhost:8000/generate"  # Change this to your actual API endpoint

def format_message(message_content):
    """Format message content for better display"""
    # Find and format course information sections
    content = message_content
    
    # Format "Course:" sections
    content = re.sub(r'(Course|Department|Professor|Schedule|Credits|Location):\s*([^\n]+)', 
                     r'<b>\1:</b> \2', 
                     content)
    
    # Format section headers
    content = re.sub(r'(COURSE DESCRIPTION|PREREQUISITES|COURSE OBJECTIVES|REQUIRED MATERIALS):', 
                     r'<h4 style="color: #041E42;">\1</h4>', 
                     content)
    
    # Replace newlines with HTML breaks
    content = content.replace('\n\n', '<br><br>').replace('\n', '<br>')
    
    return content

def stream_response(question):
    """Make API request and stream the response"""
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Create a placeholder for the streaming response
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        # Make the API request
        payload = {
            "question": question,
            "thread_id": st.session_state.thread_id
        }
        
        with st.spinner("Searching for course information..."):
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Filter for only the AI messages
                ai_messages = [msg for msg in result["result"] if msg["role"] == "ai"]
                
                if ai_messages:
                    # Get the last AI message
                    ai_response = ai_messages[-1]["content"]
                    
                    # Simulate streaming (since the actual API doesn't stream)
                    words = ai_response.split()
                    for i in range(len(words)):
                        # Add a few words at a time to simulate streaming
                        chunk_size = min(3, len(words) - i)
                        full_response += " ".join(words[i:i+chunk_size]) + " "
                        formatted_response = format_message(full_response)
                        message_placeholder.markdown(f'<div class="assistant-message">{formatted_response}</div>', unsafe_allow_html=True)
                        time.sleep(0.05)  # Adjust speed of typing animation
                        i += chunk_size - 1
                    
                    # Add the message to the history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                else:
                    message_placeholder.markdown('<div class="system-message">No response from the assistant.</div>', unsafe_allow_html=True)
            else:
                message_placeholder.markdown(f'<div class="system-message">Error: {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
                
    except Exception as e:
        message_placeholder.markdown(f'<div class="system-message">Error: {str(e)}</div>', unsafe_allow_html=True)

# Display message history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        formatted_content = format_message(message["content"])
        st.markdown(f'<div class="assistant-message">{formatted_content}</div>', unsafe_allow_html=True)

# Query input
with st.container():
    # Create a form to properly handle input clearing
    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_input("Ask about Georgetown courses:", key="user_question", placeholder="E.g., Tell me about DSAN-5100")
        submit_button = st.form_submit_button("Ask")
        
        # Process the query when submitted
        if submit_button and user_input:
            # Use the input value but don't try to clear it by modifying session_state
            stream_response(user_input)

# Reset conversation button
if st.button("Start New Conversation"):
    st.session_state.messages = []
    st.session_state.thread_id += 1
    st.experimental_rerun()

# Footer
st.markdown("""
<div class="small-text">
<p>This agent provides information about Georgetown University courses based on publicly available data. 
Information may not be complete or up-to-date. Always verify course details with official Georgetown resources.</p>
</div>
""", unsafe_allow_html=True)