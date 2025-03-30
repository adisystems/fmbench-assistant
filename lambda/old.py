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
from botocore.config import Config
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
from langchain_aws import ChatBedrockConverse
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
# Configure botocore client with retry settings
config = Config(
    retries=dict(
        max_attempts=10,
        mode='adaptive'
    ),
    connect_timeout=5,
    read_timeout=60
)
client = boto3.client("bedrock-runtime", region_name=region, config=config)

def ensure_chat_history_table_exists():
    """Create DynamoDB table for chat history if it doesn't exist."""
    try:
        dynamodb = boto3.client('dynamodb', region_name=region)
        try:
            # Check if table exists
            table = dynamodb.describe_table(TableName=table_name)
            key_schema = table['Table']['KeySchema']
            
            # Check for single primary key named 'SessionId'
            if len(key_schema) != 1 or key_schema[0]['AttributeName'] != 'SessionId':
                logger.warning("Table exists but schema is incorrect - recreating")
                dynamodb.delete_table(TableName=table_name)
                raise dynamodb.exceptions.ResourceNotFoundException(
                    "Table must have 'SessionId' as primary key"
                )
            
            logger.info(f"DynamoDB table {table_name} exists with correct schema")
            return
        except dynamodb.exceptions.ResourceNotFoundException:
            # Create table
            logger.info(f"Creating DynamoDB table {table_name}")
            response = dynamodb.create_table(
                TableName=table_name,
                AttributeDefinitions=[
                    {
                        'AttributeName': 'SessionId',
                        'AttributeType': 'S'
                    }
                ],
                KeySchema=[
                    {
                        'AttributeName': 'SessionId',
                        'KeyType': 'HASH'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'  # On-demand capacity
            )
            
            # Wait for table to be created
            waiter = dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=table_name)
            
            # Enable TTL
            dynamodb.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl'
                }
            )
            logger.info(f"DynamoDB table {table_name} created successfully")
    except Exception as e:
        logger.error(f"Error ensuring DynamoDB table exists: {e}")
        raise

# Ensure table exists before proceeding
ensure_chat_history_table_exists()

def get_message_history(session_id: str) -> DynamoDBChatMessageHistory:
    """Initialize DynamoDB chat message history."""
    try:
        history = DynamoDBChatMessageHistory(
            table_name=table_name,
            session_id=session_id,
            boto3_session=boto3.Session(region_name=region),
            ttl=24 * 60 * 60  # 24 hour TTL
        )
        logger.info(f"Initialized chat history for session {session_id}")
        return history
    except Exception as e:
        logger.error(f"Error initializing chat history: {str(e)}")
        raise

llm = ChatBedrockConverse(
    client=client,
    model='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    temperature=0.6
)

def initialize_retriever():
    """Initialize document retriever with error handling and Lambda-compatible paths."""
    try:
        # Use /tmp directory in Lambda for document storage
        data_file = Path("/tmp/documents_1.json")
        
        # Copy documents file to /tmp if it doesn't exist
        if not data_file.exists():
            source_file = Path("data/documents_1.json")
            if source_file.exists():
                data_file.write_text(source_file.read_text())
            else:
                raise FileNotFoundError(f"Source file {source_file} not found")
        
        documents_data = json.loads(data_file.read_text())
        logger.info(f"Loaded {len(documents_data)} documents")
        
        documents = [
            Document(page_content=doc["markdown"], metadata=doc.get("metadata", {}))
            for doc in documents_data
        ]
        
        embeddings_model = BedrockEmbeddings(
            client=client,
            model_id="amazon.titan-embed-text-v1"
        )
        
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
        # Create Retriever Tool
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_dsan_docs",
            description="Searches for information about the DSAN program at Georgetown University"
        )
        
        return retriever_tool
        
    except Exception as e:
        logger.error(f"Error initializing retriever: {str(e)}", exc_info=True)
        raise

# Initialize retriever tool
try:
    retriever_tool = initialize_retriever()
    logger.info("Successfully initialized retriever tool")
except Exception as e:
    logger.error("Failed to initialize retriever tool")
    raise

system_prompt = """You are an assistant that answers questions about the DSAN program at Georgetown University.

CRITICAL: You MUST follow this EXACT format for EVERY response - no exceptions:

Question: the input question you must answer
Thought: your reasoning about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the specific input for the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: explain your reasoning for the final answer
Final Answer: your complete answer to the question

IMPORTANT RULES:
1. You MUST use the search tool to find information - never answer from general knowledge
2. You MUST use the exact format above - no direct answers without going through the steps
3. Keep responses natural but concise
4. Say 'I don't have enough information to answer that' if search results are insufficient
5. Consider chat history for context when provided

Available tools:
{tools}

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

agent_runnable = create_react_agent(
    llm=llm,
    tools=[retriever_tool],
    prompt=prompt,
)

class AgentState(BaseModel):
    """State for the agent workflow."""
    messages: List[HumanMessage]
    chat_history: List[Tuple[str, str]]
    session_id: str
    next_step_id: Optional[str] = None
    intermediate_steps: List = Field(default_factory=list)

def agent_step(state: AgentState) -> AgentState:
    """Execute one step of the agent."""
    messages = state.messages
    if messages:
        logger.info(f"Starting agent step for session {state.session_id}")
        logger.debug(f"Current messages: {messages}")
        start_time = time.time()
        # Prepare input for agent including intermediate steps
        agent_input = {
            "input": messages[0].content if messages else "",
            "chat_history": state.chat_history,
            "intermediate_steps": state.intermediate_steps
        }
        
        try:
            # Ensure agent_scratchpad is included
            if "agent_scratchpad" not in agent_input:
                agent_input["agent_scratchpad"] = ""
            
            # Invoke agent with prepared input
            try:
                action = agent_runnable.invoke(agent_input)
            except Exception as parse_error:
                logger.error(f"Output parsing error: {str(parse_error)}")
                logger.error("Failed to parse LLM output - ensuring strict format adherence")
                state.next_step_id = END
                return state
            
            if not action:
                logger.warning("Agent returned no action")
                state.next_step_id = END
                return state
                
            if isinstance(action, AgentAction):
                logger.info(f"Processing tool action: {action.tool} with input: {action.tool_input}")
                tool = {"search_dsan_docs": retriever_tool}[action.tool]
                
                tool_start = time.time()
                observation = tool.invoke(action.tool_input)
                tool_duration = time.time() - tool_start
                
                logger.info(f"Tool execution completed in {tool_duration:.2f}s")
                logger.debug(f"Tool observation: {observation}")
                
                # Add to intermediate steps
                state.intermediate_steps.append((action, str(observation)))
                
                # Re-invoke agent to process the observation
                next_action = agent_runnable.invoke({
                    "input": messages[0].content,
                    "chat_history": state.chat_history,
                    "intermediate_steps": state.intermediate_steps,
                    "agent_scratchpad": ""
                })
                
                if isinstance(next_action, AgentFinish):
                    logger.info("Agent ready with final answer - ending workflow")
                    state.next_step_id = END
                else:
                    logger.info("Agent needs another step - continuing workflow")
                    state.next_step_id = "agent"
            elif isinstance(action, AgentFinish):
                logger.info(f"Agent finished with response for session {state.session_id}")
                final_answer = action.return_values["output"]
                
                # Validate final answer format
                if "Final Answer:" not in final_answer:
                    logger.error("Agent response missing 'Final Answer:' format marker")
                    logger.debug(f"Invalid response format: {final_answer}")
                    final_answer = "I apologize, but I encountered an error processing your request. Please try asking your question again."
                else:
                    # Extract just the final answer portion
                    final_answer = final_answer.split("Final Answer:")[-1].strip()
                
                logger.info(f"Processed final answer: {final_answer[:100]}...")
                try:
                    # Add to chat history
                    state.chat_history.append((messages[0].content, final_answer))
                    
                    # Save to DynamoDB
                    message_history = get_message_history(state.session_id)
                    message_history.add_user_message(messages[0].content)
                    message_history.add_ai_message(final_answer)
                    logger.info(f"Successfully saved chat history for session {state.session_id}")
                except Exception as chat_error:
                    logger.error(f"Error saving chat history: {str(chat_error)}")
                    # Continue execution even if chat history save fails
                state.next_step_id = END
            else:
                logger.warning(f"Unexpected action type: {type(action)}")
                state.next_step_id = END
        except Exception as e:
            logger.error(f"Error executing agent step: {str(e)}")
            state.next_step_id = END
            return state
        
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
            session_id=session_id,
            intermediate_steps=[]
        )
        
        # Create and execute workflow
        logger.info(f"Creating workflow for session {session_id}")
        workflow = StateGraph(AgentState)
        
        # Add agent node
        workflow.add_node("agent", agent_step)
        
        # Define edges: agent node can either continue to itself or reach END
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", "agent")
        workflow.add_edge("agent", END)
        
        # Compile workflow
        chain = workflow.compile()
        
        # Execute workflow
        logger.info(f"Executing workflow for session {session_id}")
        final_state = chain.invoke(state)
        
        # Extract answer from the most recent interaction
        logger.info("Extracting answer from final state")
        # If we have chat history, get the last AI message, otherwise return default message
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

