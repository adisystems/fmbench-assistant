import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from colorama import init, Fore
from botocore.config import Config
from langchain_core.tools import tool
from utils import create_bedrock_client
from fmbench_rag_setup import FMBenchRagSetup
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from langchain_aws import ChatBedrockConverse
from fastapi.responses import RedirectResponse
from guardrails import BedrockGuardrailManager
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage


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
_guardrail_id = None
_guardrail_version = None
_bedrock_client = None

# ----------------------------
# Tool Definition
# ----------------------------
@tool
def get_fmbench_info(
    question: str
) -> str:
    """
    Retrieves information about AWS Foundation Model Benchmarking Tool (FMBench) from official documentation.
    Use this tool for questions about benchmarking, configurations, supported models, and deployment details.
    
    Args:
        question: A clear, specific question about AWS FMBench capabilities,
                 such as supported instance types, inference containers, metrics, or deployment options.
                 
    Returns:
        A string containing the answer and additional context from the documentation.
    """
    global _rag_system
    
    # Initialize the RAG system if it hasn't been set up yet
    if _rag_system is None:
        bedrock_role_arn = os.environ.get("BEDROCK_ROLE_ARN")
        _rag_system = FMBenchRagSetup(bedrock_role_arn=bedrock_role_arn).setup()
        
    # Use the RAG system to answer the question
    result = _rag_system.query(question)
    return result["answer"]

tools = [get_fmbench_info]

# ----------------------------
# Agent Setup
# ----------------------------
SYSTEM_PROMPT = (
    "You are a helpful technical assistant for the AWS Foundation Model Benchmarking Tool (FMBench). "
    "Use the available tools to answer user questions about model benchmarking, configurations, and deployment options."
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
app = FastAPI(title="AWS Foundation Model Benchmarking Tool (FMBench) Assistant", root_path="/prod")

@app.post("/generate")
async def generate_answer(request: GenerateRequest):
    """
    Generate an answer using ReAct agent with chat history.
    
    This endpoint processes natural language questions and returns AI-generated responses.
    It maintains conversation history using thread_id and leverages AWS Bedrock models.
    """
    global _react_agent
    global _guardrail_id
    global _guardrail_version
    global _bedrock_client
    
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

        # create guardrails if not created already 
        bedrock_role_arn = os.environ.get("BEDROCK_ROLE_ARN")
        logger.info(f"bedrock_role_arn={bedrock_role_arn}")
        if _guardrail_id is None or _guardrail_version is None:
            # Basic usage with default configuration
            manager = BedrockGuardrailManager(region="us-east-1", bedrock_role_arn=bedrock_role_arn)
            _guardrail_id, _guardrail_version = manager.get_or_create_guardrail()
        
        guardrail_config = {
            "guardrailIdentifier": _guardrail_id,
            "guardrailVersion": _guardrail_version,
            "trace": "enabled"
        }
        logger.info(f"guardrail_config={guardrail_config}")

        if _bedrock_client is None:
            _bedrock_client = create_bedrock_client(bedrock_role_arn, "bedrock-runtime", region)
        model = ChatBedrockConverse(
            client=_bedrock_client,
            model=model_id,
            guardrail_config=guardrail_config,
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
        print("AWS FMBench Assistant app loaded successfully")
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