import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import boto3
from colorama import init, Fore, Style

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
app = FastAPI(title="Solar Panel AI Agent", 
              description="Customer support agent for Solar Panels Belgium")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# ----------------------------
# Solar Panel Savings Tool
# ----------------------------
from langchain_core.tools import tool

@tool
def compute_savings(monthly_cost: float) -> dict:
    """
    Compute potential savings for switching to solar energy given a monthly electricity cost.
    
    Args:
        monthly_cost (float): The user's current monthly electricity cost.
        
    Returns:
        dict: Contains:
            - 'number_of_panels': Estimated number of panels required.
            - 'installation_cost': Estimated cost for installation.
            - 'net_savings_10_years': Net savings over 10 years.
    """
    cost_per_kWh = 0.28  
    cost_per_watt = 1.50  
    sunlight_hours_per_day = 3.5  
    panel_wattage = 350  
    system_lifetime_years = 10  

    monthly_consumption_kWh = monthly_cost / cost_per_kWh    
    daily_energy_production = monthly_consumption_kWh / 30
    system_size_kW = daily_energy_production / sunlight_hours_per_day    
    number_of_panels = system_size_kW * 1000 / panel_wattage
    installation_cost = system_size_kW * 1000 * cost_per_watt    
    annual_savings = monthly_cost * 12
    total_savings_10_years = annual_savings * system_lifetime_years
    net_savings = total_savings_10_years - installation_cost
    
    return {
        "number_of_panels": round(number_of_panels),
        "installation_cost": round(installation_cost, 2),
        "net_savings_10_years": round(net_savings, 2)
    }

# ----------------------------
# Agent Setup using Latest Structure
# ----------------------------
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse

# Define system prompt
SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Solar Panels Belgium. "
    "Your goal is to extract the user's monthly electricity cost and, when provided, "
    "calculate potential savings using the compute_savings tool. "
    "If the cost is missing, ask for clarification."
)

# Tools available to the agent
tools = [compute_savings]

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
        
        # Initialize the model and client for each request
        # This ensures fresh credentials and prevents timeout issues
        bedrock_client = get_bedrock_client()
        model = ChatBedrockConverse(
            client=bedrock_client,
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            temperature=0,
            max_tokens=500
        )
        
        # Create the agent with the model and tools
        agent_executor = create_react_agent(model, tools)
        
        # Build messages with system message at the start if thread is new
        messages = conversation_memory[request.thread_id]
        if not messages:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        
        # Add the new user message
        messages.append(HumanMessage(content=request.question))
        
        # Invoke the agent
        response = agent_executor.invoke({"messages": messages})
        logger.info(response["messages"])
        
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
    logger.info("Starting Solar Panel AI Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)