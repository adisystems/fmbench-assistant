import os
import json
import boto3
import logging
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Load environment variables with explicit path 
env_path = Path(__file__).resolve().parent.parent / '.env'
try:
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    logging.error("Error loading .env file: " + str(e))

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
# AWS & DSAN RAG Setup
# ----------------------------
region = "us-west-2"
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

# Initialize the Bedrock LLM 
from langchain_aws import ChatBedrockConverse
llm = ChatBedrockConverse(client=bedrock_client, model="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

# Load DSAN documents from the JSON file in the data folder 
data_file = Path(__file__).resolve().parent.parent / "data" / "documents_1.json"
documents_data = json.loads(data_file.read_text())
logger.info(f"Loaded {len(documents_data)} documents from {data_file}")

# Convert JSON objects into LangChain Document objects
from langchain.schema import Document
documents = [
    Document(page_content=doc["markdown"], metadata=doc.get("metadata", {}))
    for doc in documents_data
]

# Split large documents into chunks for better processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = splitter.split_documents(documents)
logger.info(f"Created {len(doc_chunks)} document chunks.")

# Create embeddings using Amazon Titan
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
embeddings_model = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

# Build a FAISS vector store from the document chunks
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings_model)

# ----------------------------
# Retriever Setup
# ----------------------------
# Use the vectorstore retriever directly for efficient retrieval
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ----------------------------
# Prompt & Retrieval Chain Setup
# ----------------------------
from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant that answers questions about the DSAN program at Georgetown University. "
    "Use the provided context to answer the question concisely. If you don't have enough information, say 'I don't know'.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

from langchain.chains.combine_documents import create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# FastAPI App Setup
# ----------------------------
app = FastAPI(title="Georgetown DSAN Program Information Agent")

class QuestionRequest(BaseModel):
    question: str

@app.post("/generate")
async def generate_answer(request: QuestionRequest):
    human_message = f"[Human] {request.question}"
    logger.info(Fore.GREEN + human_message)
    results = rag_chain.invoke({"input": request.question})
    ai_message = f"[AI] {results['answer']}"
    logger.info(Fore.BLUE + ai_message)
    return {"answer": results["answer"]}

# Run the FastAPI App
if __name__ == "__main__":
    logger.info("Starting Georgetown DSAN Program Information Agent server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)