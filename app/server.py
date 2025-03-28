import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import boto3
from colorama import init, Fore, Style
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables with explicit path 
env_path = Path(__file__).parent.parent / '.env'
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
app = FastAPI(
    title="Georgetown Course Information Agent", 
    description="AI assistant for finding Georgetown University course information"
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# ----------------------------
# Unified Search Wrapper Functions
# ----------------------------
def score_results(results):
    """
    Simple heuristic scoring function based on keyword occurrence.
    Adjust or expand this as needed.
    """
    score = 0
    if isinstance(results, list):
        for result in results:
            content = ""
            if isinstance(result, dict):
                content = result.get("content", "").lower()
            else:
                content = str(result).lower()
            if "georgetown" in content:
                score += 1
            if "course" in content:
                score += 1
    return score

def unified_search(query, tavily_api_key):
    """
    Executes the query using both DuckDuckGo and Tavily,
    then returns the result with the higher heuristic score.
    """
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.tools.tavily_search import TavilySearchResults

    ddg_tool = DuckDuckGoSearchRun()
    tavily_tool = TavilySearchResults(
        max_results=5, 
        tavily_api_key=tavily_api_key, 
        search_depth="advanced"
    )

    ddg_results, tavily_results = None, None

    try:
        ddg_results = ddg_tool.invoke(query)
    except Exception as e:
        logger.warning(f"DDG search failed for query '{query}': {e}")
    
    try:
        tavily_results = tavily_tool.invoke(query)
    except Exception as e:
        logger.warning(f"Tavily search failed for query '{query}': {e}")

    ddg_score = score_results(ddg_results) if ddg_results else 0
    tavily_score = score_results(tavily_results) if tavily_results else 0

    if tavily_score > ddg_score:
        return tavily_results
    else:
        return ddg_results

def flatten_results(results):
    """
    Recursively flattens nested lists of results.
    """
    flattened = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, list):
                flattened.extend(flatten_results(item))
            else:
                flattened.append(item)
    else:
        flattened.append(results)
    return flattened

# ----------------------------
# Georgetown Course Information Tool
# ----------------------------
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults  # for typing/reference

@tool
def get_course_info(course_code: str) -> dict:
    """
    Get comprehensive information about a Georgetown University course based on its code using web search.
    """
    # Standardize course code format
    course_code = course_code.upper().replace(" ", "-")
    
    # Ensure we have an API key for Tavily
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
        
    # Generate multiple targeted search queries to get comprehensive information
    search_queries = [
        f"Georgetown University {course_code} syllabus description",
        f"Georgetown {course_code} professor instructor",
        f"Georgetown {extract_department(course_code)} {course_code} schedule semester",
        f"Georgetown course catalog {course_code}"
    ]
    
    all_results = []
    
    try:
        for query in search_queries:
            try:
                results = unified_search(query, tavily_api_key)
                all_results.append(results)
                # Brief pause to avoid overwhelming the APIs
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error during unified search with query '{query}': {e}")
                continue
        
        # Fallback to a more general query if no results were obtained
        if not all_results:
            general_query = f"Georgetown University course {course_code}"
            results = unified_search(general_query, tavily_api_key)
            all_results.append(results)
            
    except Exception as e:
        logger.error(f"Error during all unified searches: {e}")
        return {
            "course_info": {
                "course_code": course_code,
                "department": extract_department(course_code),
                "source_urls": []
            },
            "error": str(e)
        }
    
    # Flatten the results to handle nested lists
    flattened_results = []
    for res in all_results:
        flattened_results.extend(flatten_results(res))
    
    print(f"flattened_results=\n{flattened_results}")
    
    # Safely extract source URLs if available
    source_urls = []
    for result in flattened_results[:5]:
        if isinstance(result, dict) and "url" in result:
            source_urls.append(result["url"])
    
    basic_info = {
        "course_code": course_code,
        "department": extract_department(course_code),
        "source_urls": list(set(source_urls))
    }
    
    # Concatenate all content from results, ensuring each item is a string
    all_content = "\n".join(
        [result.get("content", "") if isinstance(result, dict) else str(result) for result in flattened_results]
    )
    
    return {
        "course_info": basic_info,
        "raw_search_results": flattened_results[:2],  # Limit raw results to keep response size manageable
        "all_content": all_content[:10000]  # Limit content length but provide substantial text for analysis
    }

def extract_department(course_code):
    """Extract department from course code (assumes format like COMP-167 or GOVT-020)"""
    if "-" in course_code:
        return course_code.split("-")[0]
    return course_code.split()[0] if " " in course_code else course_code

# ----------------------------
# Agent Setup using Latest Structure
# ----------------------------
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse

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
            logger.error(f"Error initializing Bedrock model: {e}")
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
        retry_delay = 3  # initial delay in seconds
        for attempt in range(max_retries):
            try:
                response = agent_executor.invoke({"messages": messages})
                logger.info(response["messages"])
                break  # successful execution, exit loop
            except Exception as e:
                if "ThrottlingException" in str(e):
                    logger.warning(f"ThrottlingException encountered on attempt {attempt+1} of {max_retries}: {e}")
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
                    logger.error(f"Error during agent execution: {e}")
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
        logger.error(f"Error in agent processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Run the FastAPI App
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Georgetown Course Information Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)