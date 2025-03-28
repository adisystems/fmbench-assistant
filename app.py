import streamlit as st
import requests
import json
import time

# Backend API URL - adjust to match your server port
API_URL = "http://localhost:8000/generate"

# Set up the page config with a dark theme and Georgetown branding
st.set_page_config(
    page_title="Georgetown DSAN Course Agent", 
    page_icon="🐶",  # Bulldog emoji as Hoya mascot
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        .disclaimer {
            font-size: 0.85rem;
            margin-top: 0.5rem;
            color: #A0A0A0;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 0.5rem;
        }
        .stButton>button {
            background-color: #333333;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            margin-top: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #555555;
        }
        .response-box {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .new-conversation-btn {
            background-color: #0056D6;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            margin-top: 1rem;
        }
        .new-conversation-btn:hover {
            background-color: #0044AA;
        }
        .history-item {
            background-color: #222222;
            padding: 8px;
            border-radius: 5px;
            margin-bottom: 5px;
        }
        .history-item:hover {
            background-color: #333333;
        }
        .user-query {
            color: #CCCCCC;
            font-weight: bold;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1
if "has_input" not in st.session_state:
    st.session_state.has_input = False

# Create a two-column layout
left_col, right_col = st.columns([1, 3])

# Sidebar with conversation history (only shown after first input)
with left_col:
    if st.session_state.has_input and len(st.session_state.conversation) > 0:
        st.markdown("### Conversation History")
        
        # Display past conversations as expandable sections
        for i in range(0, len(st.session_state.conversation), 2):
            if i+1 < len(st.session_state.conversation):
                # Get user question and truncate if too long
                user_q = st.session_state.conversation[i]["content"]
                short_q = user_q[:25] + "..." if len(user_q) > 25 else user_q
                
                # Create expandable section
                with st.expander(f"Q: {short_q}"):
                    st.markdown(f"**User:** {user_q}")
                    
                    # Show condensed answer
                    answer = st.session_state.conversation[i+1]["content"]
                    short_answer = answer[:100] + "..." if len(answer) > 100 else answer
                    st.markdown(f"**Assistant:** {short_answer}")

# Main content area
with right_col:
    # Title and introduction
    st.markdown('<div class="title">Georgetown DSAN Course Agent</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">Welcome to the Georgetown DSAN Course Agent! 🦉</div>
    <p>Ask questions about the Data Science and Analytics (DSAN) program at Georgetown University, including courses, professors, schedules, and more!</p>
    <div class="disclaimer">
    This application provides information about the Data Science and Analytics (DSAN) program at Georgetown University. Data is based on available resources and may not be fully up-to-date. Always verify details on the official Georgetown DSAN website.
    </div>
    """, unsafe_allow_html=True)
    
    # Input box for user query
    question = st.text_input("Ask about a DSAN course or program:", placeholder="e.g., What is DSAN 6725 about?", key="user_input")
    
    # Submit button
    if st.button("Submit"):
        if question.strip():
            # Set has_input to true to show conversation history
            st.session_state.has_input = True
            
            # Save the user input to conversation history
            st.session_state.conversation.append({"role": "user", "content": question})
            
            # Show a spinner while waiting for response
            with st.spinner("Getting information..."):
                # Prepare the payload
                payload = {
                    "question": question,
                    "thread_id": st.session_state.thread_id
                }
                
                try:
                    # Make the API request
                    response = requests.post(API_URL, json=payload)
                    
                    if response.status_code == 200:
                        # Process the response
                        data = response.json()
                        result = data.get("result", [])
                        
                        response_text = ""
                        for entry in result:
                            title = entry.get("title", "")
                            content = entry.get("content", "")
                            url = entry.get("url", "")
                            
                            # Format the response with Markdown
                            response_text += f"### {title}\n\n{content}\n\n"
                            if url:
                                response_text += f"[Source]({url})\n\n"
                            response_text += "---\n\n"
                        
                        # Save the assistant's response to conversation history
                        st.session_state.conversation.append({"role": "assistant", "content": response_text})
                        
                        # Force a rerun to update the UI
                        st.rerun()
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question before submitting.")
    
    # Button to start a new conversation
    if st.button("Start New Conversation", key="new_convo"):
        st.session_state.conversation = []
        st.session_state.thread_id += 1
        st.rerun()
    
    # Display current conversation with responses below questions
    for i in range(0, len(st.session_state.conversation), 2):
        if i < len(st.session_state.conversation):
            # Display user message
            user_message = st.session_state.conversation[i]
            st.markdown(f'<div class="user-query">You: {user_message["content"]}</div>', unsafe_allow_html=True)
            
            # Display assistant response (if it exists)
            if i+1 < len(st.session_state.conversation):
                assistant_message = st.session_state.conversation[i+1]
                st.markdown(f'<div class="response-box">{assistant_message["content"]}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="disclaimer">
<p>This agent provides information about Georgetown University DSAN courses based on publicly available data. 
Always verify course details with official Georgetown resources.</p>
</div>
""", unsafe_allow_html=True)