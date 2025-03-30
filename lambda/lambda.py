import os
import time
import sys
import json
import boto3
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from colorama import init, Fore, Style
from dotenv import load_dotenv
from starlette.requests import Request
from mangum import Mangum

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

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------
# FastAPI App Initialization
# ----------------------------
app = FastAPI(title="Georgetown DSAN Program Information Agent")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# ----------------------------
# Load Environment Variables
# ----------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
try:
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    logging.error("Error loading .env file: " + str(e))

# ----------------------------
# DSAN RAG Setup
# ----------------------------
region = "us-west-2"
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

from langchain_aws import ChatBedrockConverse
llm = ChatBedrockConverse(client=bedrock_client, model="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

# Load documents
data_file = Path("data/documents_1.json")
documents_data = json.loads(data_file.read_text())
logger.info(f"Loaded {len(documents_data)} documents from {data_file}")

from langchain.schema import Document
documents = [Document(page_content=doc["markdown"], metadata=doc.get("metadata", {})) for doc in documents_data]

from langchain_aws.embeddings.bedrock import BedrockEmbeddings
embeddings_model = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant that answers questions about the DSAN program at Georgetown University. "
    "Use the provided context to answer the question concisely. If you don't know, say 'I don't know'.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

from langchain.chains.combine_documents import create_stuff_documents_chain
qa_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ----------------------------
# Tool Definition
# ----------------------------
from langchain_core.tools import tool

@tool
def get_info(question: str) -> str:
    """
    Answer questions using Georgetown DSAN documentation.
    """
    result = rag_chain.invoke({"input": question})
    return result["answer"]

tools = [get_info]

# ----------------------------
# Agent Setup
# ----------------------------
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a helpful academic assistant for Georgetown University's DSAN program. "
    "Use the available tools to answer user questions about the program."
)

conversation_memory = {}

# ----------------------------
# Request Model
# ----------------------------
class QuestionRequest(BaseModel):
    question: str
    thread_id: int = 0  # Default thread_id if not provided

# ----------------------------
# Agent Endpoint
# ----------------------------
@app.post("/generate")
async def generate_answer(request: Request):
    """Generate an answer using ReAct agent with chat history."""
    # start_time = time.time()
    # context = request.scope.get("aws.context")
    # request_id = context.aws_request_id if context else "local"
    
    # try:
    #     # Get the raw body from the Lambda event
    #     body = await request.json()
    #     logger.info(f"Request received - ID: {request_id}")
    #     logger.debug(f"Request body: {body}")
        
    #     # Parse request body
    #     question = body.get('question')
    #     session_id = body.get('session_id')
        
    #     if not question or not session_id:
    #         logger.warning(f"Invalid request - missing parameters. Session: {session_id}")
    #         return {
    #             'statusCode': 400,
    #             'body': json.dumps({'error': 'Question and session_id are required'})
    #         }

    #     # Initialize agent state
    #     logger.info(f"Initializing agent state for session {session_id}")
    #     message_history = get_message_history(session_id)
    #     chat_history = [(msg.content, msg.content) for msg in message_history.messages]
    #     logger.debug(f"Chat history loaded: {len(chat_history)} messages")
        
    #     agent_executor = create_react_agent(llm, tools)
        
        
    #     messages = conversation_memory[thread_id]
    #     if not messages:
    #         messages.append(SystemMessage(content=SYSTEM_PROMPT))
    #     messages.append(HumanMessage(content=question))

    #     response = agent_executor.invoke({"messages": messages})
    #     logger.info(response["messages"])
    #     conversation_memory[thread_id] = response["messages"]

    #     outputs = [
    #         {
    #             "role": msg.__class__.__name__.lower().replace("message", ""),
    #             "content": msg.content
    #         }
    #         for msg in response["messages"]
    #     ]
    #     return {"result": outputs}

    
    # except json.JSONDecodeError as e:
    #     logger.error(f"JSON decode error - ID: {request_id}, Error: {str(e)}")
    #     return {
    #         'statusCode': 400,
    #         'body': json.dumps({'error': 'Invalid JSON in request body'})
    #     }
    # except Exception as e:
    #     logger.error(
    #         f"Internal error - ID: {request_id}, Type: {type(e).__name__}, "
    #         f"Error: {str(e)}",
    #         exc_info=True
    #     )
    #     return {
    #         'statusCode': 500,
    #         'body': json.dumps({'error': f'Internal server error: {str(e)}'})
    #     }
    logger.info(f"Received request: {request}")
    try:
        body = await request.json()
        logger.debug(f"Request body: {body}")
        
        # Parse request body
        question = body.get('question')
        session_id = body.get('session_id')
        thread_id = body.get('thread_id', 0)  # Default thread_id if not provided
        if thread_id not in conversation_memory:
            conversation_memory[thread_id] = []
        
        bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        model = ChatBedrockConverse(
            client=bedrock_client,
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        agent_executor = create_react_agent(model, tools)

        messages = conversation_memory[thread_id]
        if not messages:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        messages.append(HumanMessage(content=question))

        response = agent_executor.invoke({"messages": messages})
        logger.info(response["messages"])
        conversation_memory[thread_id] = response["messages"]

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


# Lambda Handler
handler = Mangum(
    app,
    lifespan="auto",
    api_gateway_base_path="/",
    text_mime_types=["application/json"]
)

