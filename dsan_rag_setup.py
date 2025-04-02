import os
import json
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv
from botocore.config import Config
from pydantic import BaseModel, Field
from langchain.schema import Document
from colorama import init, Fore, Style
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrockConverse
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from botocore.session import get_session
from botocore.credentials import RefreshableCredentials

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
    "%(asctime)s.%(msecs)03d,%(levelname)s,p%(process)d,%(filename)s,%(lineno)d,%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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
    response_model_id: str = Field(default="us.amazon.nova-micro-v1:0", description="Bedrock model ID to use") # us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    embedding_model_id: str = Field(default="amazon.titan-embed-text-v1", description="Amazon Bedrock embedding model to use")
    retriever_k: int = Field(default=5, description="Number of documents to retrieve")
    vector_db_path: Optional[str] = Field(default=os.path.join("indexes", "dsan_index"), description="Path to load/save FAISS vector database")
    bedrock_role_arn: Optional[str] = Field(default=None, description="ARN of the IAM role to assume for Bedrock cross-account access")
    
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
    
    def __init__(self, **data):
        super().__init__(**data)
        self.setup_logger()
        
        # Initialize Bedrock client if not provided
        if self.bedrock_client is None:
            self.bedrock_client = self._create_bedrock_client()
            self.logger.info("Bedrock client initialized")
    
    def _create_bedrock_client(self):
        """Create a Bedrock client, optionally with cross-account role assumption"""
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        
        # If a role ARN is provided, use cross-account access
        if self.bedrock_role_arn:
            self.logger.info(f"Initializing Bedrock client with cross-account role: {self.bedrock_role_arn}")
            
            def get_credentials():
                sts_client = boto3.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.bedrock_role_arn,
                    RoleSessionName='bedrock-cross-account-session',
                    # Don't set DurationSeconds when role chaining
                )
                return {
                    'access_key': assumed_role['Credentials']['AccessKeyId'],
                    'secret_key': assumed_role['Credentials']['SecretAccessKey'],
                    'token': assumed_role['Credentials']['SessionToken'],
                    'expiry_time': assumed_role['Credentials']['Expiration'].isoformat()
                }

            session = get_session()
            refresh_creds = RefreshableCredentials.create_from_metadata(
                metadata=get_credentials(),
                refresh_using=get_credentials,
                method='sts-assume-role'
            )

            # Create a new session with refreshable credentials
            session._credentials = refresh_creds
            boto3_session = boto3.Session(botocore_session=session)
            
            return boto3_session.client("bedrock-runtime", region_name=self.region, config=config)
        else:
            self.logger.info(f"Initializing Bedrock client for region: {self.region}")
            return boto3.client("bedrock-runtime", region_name=self.region, config=config)
    
    def setup_logger(self):
        """Set up the logger with proper formatting"""
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Custom formatter with all requested fields separated by commas
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d,%(levelname)s,p%(process)d,%(filename)s,%(lineno)d,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        return self.logger
        
    def setup(self):
        """Set up the RAG system with all components"""
        # Initialize the LLM
        self.llm = ChatBedrockConverse(
            client=self.bedrock_client, 
            model=self.response_model_id
        )
        
        # Initialize embeddings model
        embeddings_model = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id=self.embedding_model_id
        )
        
        # Check if we should load an existing vector store
        if self.vector_db_path and os.path.exists(self.vector_db_path):
            self.logger.info(f"Loading vector store from {self.vector_db_path}")
            self.vectorstore = FAISS.load_local(self.vector_db_path, embeddings_model, allow_dangerous_deserialization=True)
            self.logger.info(f"Successfully loaded vector store from {self.vector_db_path}")
        else:
            self.logger.info(f"vector store path {self.vector_db_path} does not exist")
            # Load documents and create vector store from scratch
            documents_data = json.loads(self.data_file_path.read_text())
            self.logger.info(f"Loaded {len(documents_data)} documents from {self.data_file_path}")
            
            # Convert to Document objects
            self.documents = [
                Document(
                    page_content=doc["markdown"], 
                    metadata=doc.get("metadata", {})
                ) for doc in documents_data
            ]
            
            # Create vector store
            self.logger.info(f"Creating new vector store with {len(self.documents)} documents")
            self.vectorstore = FAISS.from_documents(
                documents=self.documents, 
                embedding=embeddings_model
            )
            
            # Save vector store if path is specified
            if self.vector_db_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.vector_db_path)), exist_ok=True)
                self.logger.info(f"Saving vector store to {self.vector_db_path}")
                self.vectorstore.save_local(self.vector_db_path)
        
        # Create retriever
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
    
    def create_index(self):
        """Create a vector index from documents and save it to the specified path"""
        if not self.vector_db_path:
            raise ValueError("vector_db_path must be set to create and save an index")
        
        # Initialize embeddings model
        embeddings_model = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id=self.embedding_model_id
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
        
        # Create vector store and save it
        self.logger.info(f"Creating vector store with {len(self.documents)} documents")
        self.vectorstore = FAISS.from_documents(
            documents=self.documents, 
            embedding=embeddings_model
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.vector_db_path)), exist_ok=True)
        
        self.logger.info(f"Saving vector store to {self.vector_db_path}")
        self.vectorstore.save_local(self.vector_db_path)
        
        self.logger.info(f"Vector index created and saved to {self.vector_db_path}")
        return self
    
    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the RAG system"""
        if not self.rag_chain:
            self.logger.warning("RAG chain not initialized, running setup first")
            self.setup()
            
        self.logger.info(f"Processing query: {question}")
        result = self.rag_chain.invoke({"input": question})
        from pprint import pprint
        pprint(result)
        return result