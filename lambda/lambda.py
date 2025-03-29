"""
Lambda function for Georgetown DSAN Program Information RAG Agent.
Provides chat-based Q&A using AWS Bedrock, DynamoDB for persistence, and LangGraph for orchestration.
"""

# Standard Library
import os
import json
import time
import logging
import boto3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field
from starlette.requests import Request

# LangChain & LangGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# Vector Store & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import (
    DynamoDBChatMessageHistory,
)
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings

from langgraph.graph import StateGraph
from langgraph.graph import END


# FastAPI & AWS Lambda
from fastapi import FastAPI
from mangum import Mangum
from langchain_core.documents import Document

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add handler if running in Lambda
if logger.handlers:
    # AWS Lambda adds a handler by default
    logger.handlers[0].setFormatter(formatter)
else:
    # Add stream handler for local development
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# AWS Clients
region = "us-east-1"
table_name = "dsan-chat-history"
client = boto3.client("bedrock-runtime", region_name='us-east-1')

def get_message_history(session_id: str) -> DynamoDBChatMessageHistory:
    """Initialize DynamoDB chat message history."""
    return DynamoDBChatMessageHistory(
        table_name=table_name,
        session_id=session_id,
        boto3_session=boto3.Session(region_name=region),
        ttl=24 * 60 * 60,  # 24 hour TTL
        history_size=100  # Keep last 100 messages
    )

llm = ChatBedrock(
    client=client,
    model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"]
    }
)

data_file = Path("data/documents_1.json")
documents_data = json.loads(data_file.read_text())
logger.info(f"Loaded {len(documents_data)} documents from {data_file}")

documents = [
    Document(page_content=doc["markdown"], metadata=doc.get("metadata", {}))
    for doc in documents_data
]

# Create Document Chunks and Vector Store
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = splitter.split_documents(documents)
logger.info(f"Created {len(doc_chunks)} document chunks")

embeddings_model = BedrockEmbeddings(
    client=client,
    model_id="amazon.titan-embed-text-v1"
)
vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# Create Retriever Tool
retriever_tool = create_retriever_tool(
    retriever,
    name="search_dsan_docs",
    description="Searches for information about the DSAN program at Georgetown University"
)

system_prompt = """You are an assistant that answers questions about the DSAN program at Georgetown University.
Follow these rules:
1. Use the search tool to find relevant information.
2. Be natural and concise in your responses.
3. If you don't have enough information, say 'I don't know'.
4. Use the provided chat history for context.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: consider what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation sequence can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Chat History:
{chat_history}

Question: {input}
{agent_scratchpad}"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

agent_kwargs = {
    "system_message": system_prompt,
    "extra_tools": [retriever_tool],
}

agent_runnable = create_react_agent(llm, [retriever_tool], prompt)

class AgentState(BaseModel):
    """State for the agent workflow."""
    messages: List[HumanMessage]
    chat_history: List[Tuple[str, str]]
    session_id: str
    next_step_id: Optional[str] = None

def agent_step(state: AgentState) -> AgentState:
    """Execute one step of the agent."""
    messages = state.messages
    if messages:
        logger.info(f"Starting agent step for session {state.session_id}")
        logger.debug(f"Current messages: {messages}")
        
        start_time = time.time()
        action = agent_runnable.invoke({"messages": messages})
        
        if isinstance(action, AgentAction):
            logger.info(f"Agent action: {action.tool} with input: {action.tool_input}")
            tool = {"search_dsan_docs": retriever_tool}[action.tool]
            
            tool_start = time.time()
            observation = tool.invoke(action.tool_input)
            tool_duration = time.time() - tool_start
            
            logger.info(f"Tool execution completed in {tool_duration:.2f}s")
            logger.debug(f"Tool observation: {observation}")
            
            messages.append(HumanMessage(content=str(observation)))
            state.next_step_id = "agent"
        else:
            logger.info(f"Agent finished with response for session {state.session_id}")
            state.chat_history.append((messages[0].content, action.return_values["output"]))
            message_history = get_message_history(state.session_id)
            message_history.add_user_message(messages[0].content)
            message_history.add_ai_message(action.return_values["output"])
            state.next_step_id = "end"
        
        total_duration = time.time() - start_time
        logger.info(f"Agent step completed in {total_duration:.2f}s")
    
    return state


# FastAPI App
app = FastAPI(
    title="Georgetown DSAN Program Information Agent",
    root_path=""  # Ensure root path is empty for Lambda
)

class QuestionRequest(BaseModel):
    """Request model for generate endpoint."""
    question: str
    session_id: str

@app.get("/")
async def root():
    """Root endpoint handler."""
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Georgetown DSAN Program Information Agent API",
            "endpoints": {
                "POST /generate": "Generate answers to questions about DSAN program"
            }
        })
    }

@app.post("/generate")
async def generate_answer(request: Request) -> Dict[str, Any]:
    """Generate an answer using ReAct agent with chat history."""
    start_time = time.time()
    context = request.scope.get("aws.context")
    request_id = context.aws_request_id if context else "local"
    
    try:
        # Get the raw body from the Lambda event
        body = await request.json()
        logger.info(f"Request received - ID: {request_id}")
        logger.debug(f"Request body: {body}")
        
        # Parse request body
        question = body.get('question')
        session_id = body.get('session_id')
        
        if not question or not session_id:
            logger.warning(f"Invalid request - missing parameters. Session: {session_id}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Question and session_id are required'})
            }

        # Initialize agent state
        logger.info(f"Initializing agent state for session {session_id}")
        message_history = get_message_history(session_id)
        chat_history = [(msg.content, msg.content) for msg in message_history.messages]
        logger.debug(f"Chat history loaded: {len(chat_history)} messages")
        
        state = AgentState(
            messages=[HumanMessage(content=question)],
            chat_history=chat_history,
            session_id=session_id
        )
        
        # Execute agent workflow
        logger.info(f"Executing agent workflow for session {session_id}")
        final_state = agent_runnable.invoke(state)
        answer = final_state.chat_history[-1][1] if final_state.chat_history else "I couldn't generate an answer."
        
        # Calculate timing
        total_duration = time.time() - start_time
        logger.info(f"Request completed - ID: {request_id}, Duration: {total_duration:.2f}s")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': answer,
                'request_id': request_id,
                'session_id': session_id,
                'duration': round(total_duration, 2)
            })
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error - ID: {request_id}, Error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
    except Exception as e:
        logger.error(
            f"Internal error - ID: {request_id}, Type: {type(e).__name__}, "
            f"Error: {str(e)}",
            exc_info=True
        )
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

# Lambda Handler
handler = Mangum(
    app,
    lifespan="auto",
    api_gateway_base_path="/",
    text_mime_types=["application/json"]
)