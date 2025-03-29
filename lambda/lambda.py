"""
Lambda function for Georgetown DSAN Program Information RAG Agent.
Provides chat-based Q&A using AWS Bedrock, DynamoDB for persistence, and LangGraph for orchestration.
"""

# Standard Library
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

# LangChain & LangGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough

# Vector Store & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_aws import ChatBedrockConverse
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

# AWS & FastAPI
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from Magnum import Magnum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / '.env'
try:
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    logger.error(f"Error loading .env file: {str(e)}")

# AWS Service Setup
region = "us-east-1"
bedrock_client = boto3.client("bedrock-runtime", region_name=region)
dynamodb = boto3.resource('dynamodb', region_name=region)

# DynamoDB Setup
table_name = 'DSANChatHistory'
try:
    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
        ],
        AttributeDefinitions=[
            {'AttributeName': 'SessionId', 'AttributeType': 'S'}
        ],
        BillingMode='PAY_PER_REQUEST'
    ).wait_until_exists()
    logger.info(f"Created table {table_name}")
except dynamodb.meta.client.exceptions.ResourceInUseException:
    logger.info(f"Table {table_name} already exists")

class AgentState(BaseModel):
    """State for agent execution."""
    messages: List[Dict] = Field(default_factory=list)
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)
    session_id: str
    next_step_id: str = Field(default="agent")

def get_message_history(session_id: str) -> DynamoDBChatMessageHistory:
    """Get or create a new chat message history for the session."""
    return DynamoDBChatMessageHistory(
        table_name=table_name,
        session_id=session_id,
        boto3_session=boto3.Session(region_name=region),
        ttl=24 * 60 * 60,  # 24 hour TTL
        history_size=100  # Keep last 100 messages
    )

# Initialize LLM and Load Documents
llm = ChatBedrockConverse(
    client=bedrock_client,
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

data_file = Path(__file__).resolve().parent.parent / "data" / "documents_1.json"
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
    client=bedrock_client,
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

# Create ReAct Agent
system_prompt = """You are an assistant that answers questions about the DSAN program at Georgetown University.
Follow these rules:
1. Use the search tool to find relevant information
2. Be natural and concise in your responses
3. If you don't have enough information, say 'I don't know'
4. Use the provided chat history for context

Chat History: {chat_history}
Human: {input}
Assistant: Let's solve this step by step:"""

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

def agent_step(state: AgentState) -> AgentState:
    """Execute one step of the agent."""
    messages = state.messages
    if messages:
        action = agent_runnable.invoke({"messages": messages})
        if isinstance(action, AgentAction):
            tool = {"search_dsan_docs": retriever_tool}[action.tool]
            observation = tool.invoke(action.tool_input)
            messages.append(HumanMessage(content=str(observation)))
            state.next_step_id = "agent"
        else:
            state.chat_history.append((messages[0].content, action.return_values["output"]))
            message_history = get_message_history(state.session_id)
            message_history.add_user_message(messages[0].content)
            message_history.add_ai_message(action.return_values["output"])
            state.next_step_id = "end"
    return state

# Create LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_step)
workflow.set_entry_point("agent")
workflow.add_node("end", END)

workflow.add_edge("agent", "agent")
workflow.add_edge("agent", "end")

chain = workflow.compile()

# FastAPI App
app = FastAPI(title="Georgetown DSAN Program Information Agent")

class QuestionRequest(BaseModel):
    """Request model for generate endpoint."""
    question: str
    session_id: str

@app.post("/generate")
async def generate_answer(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Generate an answer using ReAct agent with chat history."""
    try:
        # Parse request body from event
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        question = body.get('question')
        session_id = body.get('session_id')
        
        if not question or not session_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Question and session_id are required'})
            }

        # Initialize agent state
        message_history = get_message_history(session_id)
        chat_history = [(msg.content, msg.content) for msg in message_history.messages]
        
        state = AgentState(
            messages=[HumanMessage(content=question)],
            chat_history=chat_history,
            session_id=session_id
        )
        
        # Execute agent workflow
        final_state = chain.invoke(state)
        answer = final_state.chat_history[-1][1] if final_state.chat_history else "I couldn't generate an answer."
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': answer,
                'request_id': context.aws_request_id if context else None,
                'session_id': session_id
            })
        }
    
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
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