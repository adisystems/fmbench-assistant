import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import boto3
from colorama import init, Fore, Style
import re
from dotenv import load_dotenv

# Load environment variables with explicit path
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'  # Go up one directory to find .env in project root
load_dotenv(dotenv_path=env_path)

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
app = FastAPI(title="Georgetown Course Information Agent", 
              description="AI assistant for finding Georgetown University course information")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# ----------------------------
# Georgetown Course Information Tool
# ----------------------------
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def get_course_info(course_code: str) -> dict:
    """
    Get comprehensive information about a Georgetown University course based on its code using web search.
    
    Args:
        course_code (str): The course code (e.g., COMP-167, GOVT-020, DSAN-500)
        
    Returns:
        dict: Contains basic course details and raw search results for the LLM to analyze and extract structured data.
    """
    # Standardize course code format
    course_code = course_code.upper().replace(" ", "-")
    
    # Ensure we have an API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("Tavily API key not found in environment variables")
        return {
            "course_info": {
                "course_code": course_code,
                "department": extract_department(course_code),
                "source_urls": []
            },
            "error": "Missing Tavily API key"
        }
        
    # Create Tavily search tool with explicit API key and more results
    from langchain_community.tools import DuckDuckGoSearchRun

    search_tool = DuckDuckGoSearchRun()
    #search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key, search_depth="advanced")
    
    # Generate multiple targeted search queries to get comprehensive information
    search_queries = [
        f"Georgetown University {course_code} syllabus description",
        f"Georgetown {course_code} professor instructor",
        f"Georgetown {extract_department(course_code)} {course_code} schedule semester",
        f"Georgetown course catalog {course_code}"
    ]
    
    all_results = []
    
    # Execute search with error handling, trying multiple queries
    try:
        for query in search_queries:
            try:
                search_results = search_tool.invoke(query)
                all_results.append(search_results)
                # Brief pause to avoid overwhelming the API
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error during Tavily search with query '{query}': {str(e)}")
                continue
        
        # If we got no results at all, try a more general query
        if not all_results:
            general_query = f"Georgetown University course {course_code}"
            search_results = search_tool.invoke(general_query)
            all_results.append(search_results)
            
    except Exception as e:
        logger.error(f"Error during all Tavily searches: {str(e)}")
        return {
            "course_info": {
                "course_code": course_code,
                "department": extract_department(course_code),
                "source_urls": []
            },
            "error": str(e)
        }
    
    print(f"all_results=\n{all_results}")
    
    # Safely extract source URLs if available
    source_urls = []
    for result in all_results[:5]:
        if isinstance(result, dict) and "url" in result:
            source_urls.append(result["url"])
    
    basic_info = {
        "course_code": course_code,
        "department": extract_department(course_code),
        "source_urls": list(set(source_urls))
    }
    
    # Safely extract all content from results, handling both dicts and strings
    all_content = "\n".join(
        [result.get("content", "") if isinstance(result, dict) else result for result in all_results]
    )
    
    return {
        "course_info": basic_info,
        "raw_search_results": all_results[:2],  # Limit raw results to keep response size manageable
        "all_content": all_content[:10000]  # Limit content length but provide substantial text for analysis
    }

def extract_department(course_code):
    """Extract department from course code"""
    # Assumes format like COMP-167, GOVT-020
    if "-" in course_code:
        return course_code.split("-")[0]
    return course_code.split()[0] if " " in course_code else course_code

# ----------------------------
# Agent Setup using Latest Structure
# ----------------------------
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse

# Define system prompt with updated instruction regarding extraction
SYSTEM_PROMPT = """
You are a helpful AI assistant specialized in providing comprehensive information about Georgetown University courses.

Your main capability is accessing information about any Georgetown course by using web search.
When students ask about courses, extract the course code and use the get_course_info tool.

You will receive search results that include both structured information and raw content from various websites.
Your job is to carefully analyze ALL of this information to provide the most complete picture of the course.
NOTE: Do not rely on any automated regex parsing from the search tool response. Instead, analyze the raw search results provided and extract the structured course data (such as title, professor, schedule, etc.) directly in your answer.

Present your responses in a clean, simple format that will display well in any interface:

1. Start with a brief, friendly introduction.

2. Present key course information in short, labeled sections:
   Course: [DEPT-123: Full Course Title]
   Department: [Full department name]
   Professor: [Professor name(s)]
   Schedule: [Days and times]
   Credits: [Number of credits]
   Location: [Building and room if available]

3. For the course description, use a clear heading and paragraph format:
   COURSE DESCRIPTION:
   [Provide a detailed description]

4. For additional information, use simple headings and short paragraphs or single-line items:
   PREREQUISITES:
   [List prerequisites]

   COURSE OBJECTIVES:
   [List objectives]

   REQUIRED MATERIALS:
   [List materials]

5. End with helpful resources where students can find more information.

Even if information isn't explicitly structured in the returned data, analyze the raw content to find relevant details.
Be friendly, helpful, and thorough while keeping your formatting simple and universally compatible.
Always introduce yourself as the Georgetown Course Assistant in your first response.
"""

# Tools available to the agent
tools = [get_course_info]

# Initialize Bedrock client
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

# Define a memory store for conversation history
conversation_memory = {}

# ----------------------------
# Request Model for API Input
# ----------------------------
class QuestionRequest(BaseModel):
    question: str
    thread_id: int  # Used for conversation tracking

# ----------------------------
# API Endpoint to Run the Agent
# ----------------------------
@app.post("/generate")
async def generate_route(request: QuestionRequest):
    try:
        logger.info(f"Received request: {request}")
        
        # Get or initialize conversation memory for this thread
        if request.thread_id not in conversation_memory:
            conversation_memory[request.thread_id] = []
        
        # Initialize the model and client for each request with retry handling
        try:
            bedrock_client = get_bedrock_client()
            model = ChatBedrockConverse(
                client=bedrock_client,
                model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                temperature=0,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Error initializing Bedrock model: {str(e)}")
            return {
                "result": [{
                    "role": "system",
                    "content": "I'm sorry, but I'm experiencing connectivity issues with our AI service. Please try again in a few moments."
                }]
            }
        
        # Create the agent with the model and tools
        agent_executor = create_react_agent(model, tools)
        
        # Build messages with system message at the start if thread is new
        messages = conversation_memory[request.thread_id]
        if not messages:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        
        # Add the new user message
        messages.append(HumanMessage(content=request.question))
        
        # Invoke the agent with retry handling for throttling exceptions
        max_retries = 5
        retry_delay = 3  # initial delay increased to 3 seconds
        for attempt in range(max_retries):
            try:
                response = agent_executor.invoke({"messages": messages})
                logger.info(response["messages"])
                break  # successful execution, exit loop
            except Exception as e:
                if "ThrottlingException" in str(e):
                    logger.warning(f"ThrottlingException encountered on attempt {attempt+1} of {max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        return {
                            "result": [{
                                "role": "system", 
                                "content": "I'm sorry, but our service is currently experiencing high demand. Please try again in a few moments."
                            }]
                        }
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logger.error(f"Error during agent execution: {str(e)}")
                    return {
                        "result": [{
                            "role": "system",
                            "content": "I'm sorry, but I encountered an error while processing your request. Please try again later."
                        }]
                    }
        
        # Update conversation memory
        conversation_memory[request.thread_id] = response["messages"]
        
        # Return the full conversation for the client
        outputs = []
        for message in response["messages"]:
            message_type = message.__class__.__name__.lower().replace("message", "")
            outputs.append({
                "role": message_type,
                "content": message.content
            })
        
        return {"result": outputs}
        
    except Exception as e:
        logger.error(f"Error in agent processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Run the FastAPI App
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Georgetown Course Information Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)