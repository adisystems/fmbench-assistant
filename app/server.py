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


def get_or_create_guardrail(bedrock_client):
    guardrail_name = 'dsan-program-guardrails'

    try:
        # First, check if a guardrail with this name already exists
        existing_guardrails = bedrock_client.list_guardrails()['guardrails']
        logger.info(f"Existing guardrails: {existing_guardrails}")
        for guardrail in existing_guardrails:
            if guardrail['name'] == guardrail_name:
                logger.info(f"Guardrail already exists: {guardrail}")
                return guardrail['id'], guardrail['version']

        # If not found, create it
        response = bedrock_client.create_guardrail(
            name=guardrail_name,
            description='Ensures the chatbot provides accurate information based on DSAN program materials while maintaining academic integrity and appropriate boundaries.',
            topicPolicyConfig={
                'topicsConfig': [
                    # {
                    #     'name': 'Non-Public Information',
                    #     'definition': 'Providing information that is not available on public DSAN program webpages or making claims beyond official program documentation.',
                    #     'examples': [
                    #         'What are the private discussions in faculty meetings?',
                    #         'How many students were rejected last year?',
                    #         'What is the exact acceptance rate?',
                    #         'Can you share internal program metrics?',
                    #         'Who are the students currently enrolled in DSAN?'
                    #     ],
                    #     'type': 'DENY'
                    # },
                    {
                        'name': 'Future Predictions',
                        'definition': 'Making predictions or promises about future program changes, admissions, or course offerings not stated in official DSAN documentation.',
                        'examples': [
                            'Will the DSAN program requirements change next year?',
                            'What new data science courses will be added?',
                            'Will the program tuition increase next semester?',
                            'What will be the future job placement rate?',
                            'Are there plans to change the curriculum?'
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Personal Advice',
                        'definition': 'Providing personalized recommendations or decisions that should be made by DSAN program administrators or academic advisors.',
                        'examples': [
                            'Should I choose DSAN over other data science programs?',
                            'Which DSAN electives should I take given my background?',
                            'Would I be successful in the DSAN program?',
                            'Can I handle the advanced analytics coursework?',
                            'What are my career prospects after DSAN?'
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Academic Integrity',
                        'definition': 'Maintaining academic integrity by not providing direct answers to assignments, exam questions, or project solutions.',
                        'examples': [
                            'Can you solve this DSAN homework problem?',
                            'What are the answers to the midterm exam?',
                            'Write code for my class project.',
                            'Debug my assignment solution.',
                            'How should I answer this quiz question?'
                        ],
                        'type': 'DENY'
                    },
                    {
                        'name': 'Technical Implementation',
                        'definition': 'Providing specific technical implementation details about the DSAN program systems or infrastructure.',
                        'examples': [
                            'How is the DSAN website backend implemented?',
                            'What servers does the program use?',
                            'How are student records stored?',
                            'What is the database structure?',
                            'Share the system architecture details.'
                        ],
                        'type': 'DENY'
                    }
                ]
            },
            contentPolicyConfig={
                'filtersConfig': [
                    {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'INSULTS', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'MISCONDUCT', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'PROMPT_ATTACK', 'inputStrength': 'HIGH', 'outputStrength': 'NONE'}
                ]
            },
            blockedInputMessaging = (
            "It looks like your message might contain sensitive, inappropriate, or restricted content. "
            "I’m here to help within respectful and academic boundaries. For accurate information about the DSAN program, please visit https://analytics.georgetown.edu or reach out to a program advisor."
            ),
            blockedOutputsMessaging = (
            "I can’t provide that information as it may involve sensitive topics or go beyond what I'm allowed to share. "
            "For reliable details about the DSAN program, please visit https://analytics.georgetown.edu or contact a program administrator."
            ),
        )
        logger.info(f"Guardrail created successfully: {response}")
        return response['guardrailId'], response['version']

    except Exception as e:
        logger.error(f"Failed to get or create guardrail: {str(e)}")
        raise


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

        bedrock = boto3.client("bedrock", region_name=region)
        bedrock_client = boto3.client("bedrock-runtime", region_name=region, config=config)
        guardrail_id, guardrail_version = get_or_create_guardrail(bedrock)
        guardrail_config = {
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": guardrail_version,
        "trace": "enabled"
        }

        model = ChatBedrockConverse(
            client=bedrock_client,
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