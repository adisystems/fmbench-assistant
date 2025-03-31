import json
import boto3
import logging
from pathlib import Path
from mangum import Mangum
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

class DSANRagSetup(BaseModel):
    """
    Pydantic model for DSAN RAG Setup that encapsulates the entire configuration and setup process
    """
    region: str = Field(default="us-east-1", description="AWS region to use for Amazon Bedrock")
    data_file_path: Path = Field(default=Path("data/documents_1.json"), description="Path to the documents data file")
    response_model_id: str = Field(default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", description="Bedrock model ID to use")
    embedding_model_id: str = Field(default="amazon.titan-embed-text-v1", description="Amazon Bedrock embedding model to use")
    retriever_k: int = Field(default=5, description="Number of documents to retrieve")
    
    # These will be initialized in the setup method
    bedrock_client: Optional[Any] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)
    documents: List[Document] = Field(default_factory=list, exclude=True)
    vectorstore: Optional[Any] = Field(default=None, exclude=True)
    retriever: Optional[Any] = Field(default=None, exclude=True)
    rag_chain: Optional[Any] = Field(default=None, exclude=True)
    
    # Configure logger
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__), exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def setup_logger(self):
        """Set up the logger with proper formatting"""
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Custom formatter with all requested fields separated by commas
        formatter = logging.Formatter(
            "%(asctime)s,%(levelname)s,%(process)d,%(filename)s,%(lineno)d,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f"
        )
        
        # Add handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        return self.logger
        
    def setup(self):
        """Set up the RAG system with all components"""
        # Setup logger first
        self.setup_logger()
        
        # Initialize Bedrock client
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=self.region, config=config)
        
        # Initialize the LLM
        self.llm = ChatBedrockConverse(
            client=self.bedrock_client, 
            model=self.response_model_id
        )
        
        # Load documents
        documents_data = json.loads(self.data_file_path.read_text())
        self.logger.info(f"Loaded {len(documents_data)} documents from {self.data_file_path}")
        
        # Convert to Document objects
        self.documents = [
            Document(
                page_content=doc["markdown"], 
                metadata=doc.get("metadata", {})
            ) for doc in documents_data
        ]
        
        # Initialize embeddings model
        embeddings_model = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id=self.embedding_model_id
        )
        
        # Create vector store and retriever
        self.vectorstore = FAISS.from_documents(
            documents=self.documents, 
            embedding=embeddings_model
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': self.retriever_k}
        )
        
        # Create prompt template
        system_prompt = (
            "You are a friendly and helpful AI assistant that answers questions about the "
            "Data Science & Analytics (DSAN) program at Georgetown University. "
            "Use the provided context to answer the question concisely. "
            "If you don't know, say 'I don't know'.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create the chain
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, qa_chain)
        
        self.logger.info("RAG setup complete")
        return self
    
    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the RAG system"""
        if not self.rag_chain:
            self.logger.warning("RAG chain not initialized, running setup first")
            self.setup()
            
        self.logger.info(f"Processing query: {question}")
        result = self.rag_chain.invoke({"input": question})
        return result
    


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
        _rag_system = DSANRagSetup().setup()
        
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
        default="us.anthropic.claude-3-5-sonnet-20241022-v2:0", #us.amazon.nova-lite-v1:0", 
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
        thread_id = body.get('thread_id', 0)
        region = body.get('region', "us-east-1")
        model_id = body.get('response_model_id', "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        
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
    return dict(docs="something")
    #RedirectResponse("/docs")

# Lambda Handler for AWS Lambda deployment
handler = Mangum(
    app,
    lifespan="auto",
    api_gateway_base_path="/prod",
    text_mime_types=["application/json"]
)