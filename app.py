import streamlit as st
import requests
import json
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Georgetown Course Agent",
    page_icon="🐶",  # Bulldog emoji (closest to Hoya mascot)
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

# App title
st.title("Georgetown Course Agent")
st.markdown("Ask questions about Georgetown University courses, departments, and programs. You can also ask follow-up questions about courses you've already asked about.")

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = 0
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False

# API endpoint 
API_URL = "http://localhost:8000/generate"  # Change this to your actual API endpoint

def format_message(message_content):
    """Format message content for better display"""
    # Find and format course information sections
    content = message_content
    
    # Clean up repeated text that might come from streaming
    content = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', content)
    
    # Clean up section headers that might be repeated
    content = re.sub(r'(COURSE DESCRIPTION:?\s*){2,}', r'COURSE DESCRIPTION:', content, flags=re.IGNORECASE)
    content = re.sub(r'(PREREQUISITES:?\s*){2,}', r'PREREQUISITES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(ADDITIONAL INFORMATION:?\s*){2,}', r'ADDITIONAL INFORMATION:', content, flags=re.IGNORECASE)
    content = re.sub(r'(RESOURCES:?\s*){2,}', r'RESOURCES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(COURSE OBJECTIVES:?\s*){2,}', r'COURSE OBJECTIVES:', content, flags=re.IGNORECASE)
    content = re.sub(r'(REQUIRED MATERIALS:?\s*){2,}', r'REQUIRED MATERIALS:', content, flags=re.IGNORECASE)
    
    # Format "Course:" sections
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
    """Make API request and stream the response"""
    # Indicate we're awaiting a response
    st.session_state.awaiting_response = True
    
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Create a placeholder for the streaming response
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        # Add context to the question if it's a follow-up
        if len(st.session_state.messages) > 2:  # More than just the first Q&A
            # Check if the question is a short follow-up
            if len(question.split()) < 8 and not any(code in question.upper() for code in ["DSAN", "COSC", "MATH", "GOVT", "INAF"]):
                # It seems like a follow-up, so add context
                last_context = ""
                # Look for the most recent course code mentioned
                for msg in reversed(st.session_state.messages[:-1]):  # Exclude the current question
                    if msg["role"] == "user":
                        # Extract course codes from previous questions
                        codes = re.findall(r'([A-Za-z]{2,4}[-\s]?\d{3,4})', msg["content"])
                        if codes:
                            last_context = f"Regarding {codes[0]}, "
                            break
                
                # Prepend context if found
                if last_context:
                    question = last_context + question
        
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
                    for i in range(0, len(words), 3):  # Process 3 words at a time
                        # Add a few words at a time to simulate streaming
                        chunk_size = min(3, len(words) - i)
                        full_response += " ".join(words[i:i+chunk_size]) + " "
                        
                        # Only update display every few chunks to avoid duplication artifacts
                        if i % 9 == 0 or i >= len(words) - 3:
                            formatted_response = format_message(full_response)
                            message_placeholder.markdown(f'<div class="assistant-message">{formatted_response}</div>', unsafe_allow_html=True)
                            time.sleep(0.1)  # Slightly slower typing for better readability
                    
                    # Add the message to the history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                else:
                    message_placeholder.markdown('<div class="system-message">No response from the assistant.</div>', unsafe_allow_html=True)
            else:
                message_placeholder.markdown(f'<div class="system-message">Error: {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
                
    except Exception as e:
        message_placeholder.markdown(f'<div class="system-message">Error: {str(e)}</div>', unsafe_allow_html=True)
    
    # We're no longer awaiting a response
    st.session_state.awaiting_response = False
    
    # Force a rerun to update the UI with input field at the bottom
    st.rerun()

# Main app flow
def main():
    # Display message history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            formatted_content = format_message(message["content"])
            st.markdown(f'<div class="assistant-message">{formatted_content}</div>', unsafe_allow_html=True)
    
    # Add spacing to separate history from input
    st.write("")
    st.write("")
    
    # Check if we need to show input form
    if not st.session_state.awaiting_response:
        # Input form at the bottom
        with st.container():
            # Create a form to properly handle input clearing
            with st.form(key="query_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Continue the conversation:", 
                    key="user_question", 
                    placeholder="Ask about courses at Georgetown..."
                )
                col1, col2 = st.columns([4, 1])
                with col2:
                    submit_button = st.form_submit_button("Submit")
                
                # Process the query when submitted
                if submit_button and user_input:
                    # Process in the next rerun to ensure input appears after last message
                    st.session_state.pending_question = user_input
                    st.rerun()
    
    # Check if we have a pending question to process
    if 'pending_question' in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        stream_response(question)
    
    # Reset conversation button at the very bottom
    if not st.session_state.awaiting_response:
        if st.button("Start New Conversation"):
            st.session_state.messages = []
            st.session_state.thread_id += 1
            st.rerun()

# Footer
st.markdown("""
<div class="small-text">
<p>This agent provides information about Georgetown University courses based on publicly available data. 
Information may not be complete or up-to-date. Always verify course details with official Georgetown resources.</p>
</div>
""", unsafe_allow_html=True)

# Run the main function
main()