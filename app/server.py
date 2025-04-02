import os
import json
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.config import Config
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_core.tools import tool
from starlette.requests import Request
from colorama import init, Fore, Style
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrockConverse
from fastapi.responses import RedirectResponse
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from botocore.session import get_session
from botocore.credentials import RefreshableCredentials
from dsan_rag_setup import DSANRagSetup

# ----------------------------
# Setup Logging with Colorama
# ----------------------------
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = record.msg
        if isinstance(msg, list):
            formatted_messages = []
            for m in msg:
                cname = m.__class__.__name__
                if cname == 'HumanMessage':
                    formatted = f"{Fore.GREEN}[Human] {m.content}"
                elif cname == 'AIMessage':
                    formatted = f"{Fore.BLUE}[AI] {m.content}"
                elif cname == 'ToolMessage':
                    formatted = f"{Fore.YELLOW}[Tool] {m.content}"
                else:
                    formatted = str(m)
                formatted_messages.append(formatted)
            record.msg = "\n".join(formatted_messages)
        return super().format(record)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

# Custom formatter with all requested fields separated by commas
formatter = logging.Formatter(
    "%(asctime)s,%(levelname)s,%(process)d,%(filename)s,%(lineno)d,%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f"
)

# Add handler with the custom formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


# ----------------------------
# Load Environment Variables
# ----------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
try:
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    logging.error("Error loading .env file: " + str(e))


# Global instance of the RAG setup
_rag_system = None
_react_agent = None

# ----------------------------
# Tool Definition
# ----------------------------
@tool
def get_dsan_info(
    question: str
) -> str:
    """
    Retrieves information about Georgetown's Data Science & Analytics (DSAN) program from official documentation. 
    Use this tool for questions about courses, requirements, faculty, admissions, or program details.
    
    Args:
        question: A clear, specific question about Georgetown's Data Science & Analytics program,
                 such as course offerings, degree requirements, application processes, or faculty.
                 
    Returns:
        A string containing the answer and additional context from the documentation.
    """
    global _rag_system
    
    # Initialize the RAG system if it hasn't been set up yet
    if _rag_system is None:        
        _rag_system = DSANRagSetup(bedrock_role_arn="arn:aws:iam::605134468121:role/BedrockCrossAccount2").setup()
        
    # Use the RAG system to answer the question
    result = _rag_system.query(question)
    return result["answer"]

tools = [get_dsan_info]

# ----------------------------
# Agent Setup
# ----------------------------
SYSTEM_PROMPT = (
    "You are a helpful academic assistant for Georgetown University's DSAN program. "
    "Use the available tools to answer user questions about the program."
)

conversation_memory = {}
class GenerateRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    region: str = Field(default="us-east-1", description="AWS region for Bedrock")
    response_model_id: str = Field(
        default="us.amazon.nova-lite-v1:0", #us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        description="Bedrock model ID to use"
    )
    thread_id: Optional[int] = Field(
        default=0, 
        description="Conversation thread ID for maintaining chat history"
    )

class MessageOutput(BaseModel):
    role: str = Field(..., description="Role of the message sender (system, human, ai)")
    content: str = Field(..., description="Content of the message")

class GenerateResponse(BaseModel):
    result: List[MessageOutput] = Field(..., description="List of messages in the conversation")


# ----------------------------
# FastAPI App Initialization
# ----------------------------
app = FastAPI(title="Georgetown University DSAN Program Information Agent", root_path="/prod")

@app.post("/generate")
async def generate_answer(request: GenerateRequest):
    """
    Generate an answer using ReAct agent with chat history.
    
    This endpoint processes natural language questions and returns AI-generated responses.
    It maintains conversation history using thread_id and leverages AWS Bedrock models.
    """
    global _react_agent
    logger.info(f"Received request: {request}")
    try:
        body = request.model_dump()
        print(f"Request body: {body}")
        # Extract parameters from the validated request model
        question = body.get('question')
        thread_id = body.get('thread_id')
        region = body.get('region')
        model_id = body.get('response_model_id')
        
        # Initialize or retrieve conversation memory
        if thread_id not in conversation_memory:
            conversation_memory[thread_id] = []
        
        # Set up Bedrock client and model
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        bedrock_client = boto3.client("bedrock-runtime", region_name=region, config=config)
        model = ChatBedrockConverse(
            client=bedrock_client,
            model=model_id
        )
        
        # Create the agent executor
        if _react_agent is None:
            _react_agent = create_react_agent(model, tools)

        messages = conversation_memory[thread_id]
        if not messages:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        messages.append(HumanMessage(content=question))

        response = _react_agent.invoke({"messages": messages})
        logger.info(response["messages"])
        conversation_memory[request.thread_id] = response["messages"]

        # Format the output
        outputs = [
            {
                "role": msg.__class__.__name__.lower().replace("message", ""),
                "content": msg.content
            }
            for msg in response["messages"]
        ]
        return {"result": outputs}

    except Exception as e:
        logger.error(f"Error in agent processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs")
async def redirect_root_to_docs():
    RedirectResponse("/docs")

# the following check allows for the same code to work locally as well as inside a Lambda function
inside_lambda = os.environ.get("AWS_EXECUTION_ENV") is not None
if not inside_lambda:
    logger.info(f"not running inside a Lambda")
    if __name__ == "__main__":
        import uvicorn
        # ----------------------------
        # Start the FastAPI App Locally
        # ----------------------------
        # Only run the server directly when this file is executed as a script
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # When imported by langchain serve, just define the app without running it
        print("Georgetown DSAN Program Information Agent app loaded successfully")
else:
    logger.info(f"running inside a Lambda")
    # Lambda Handler for AWS Lambda deployment
    from mangum import Mangum
    handler = Mangum(
        app,
        lifespan="auto",
        api_gateway_base_path="/prod",
        text_mime_types=["application/json"]
)