import os
import json
import logging
import numpy as np
import faiss
import torch
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv

# Set threading limits for better performance
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "documents_1.json")
INDEX_PATH = os.path.join(BASE_DIR, "data", "dsan_faiss.index")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_documents.json")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "documents_mapping.json")

# ----------------------------
# System Prompt for the Agent
# ----------------------------
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

class RAGSearchTool:
    def __init__(self, index_path, data_path):
        """Initialize the FAISS-based RAG search tool with course-specific capabilities"""
        try:
            # Paths
            self.index_path = index_path
            self.data_path = data_path
            self.mapping_path = os.path.join(os.path.dirname(index_path), "documents_mapping.json")
            self.processed_data_path = os.path.join(os.path.dirname(data_path), "processed_documents.json")
            
            # Load FAISS index
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded successfully.")
            
            # Load tokenizer and model for embedding generation
            logger.info("Loading tokenizer and model for embedding generation")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cpu")
            
            # Load transformer for text generation/summarization
            logger.info("Loading text generation model")
            try:
                self.text_generator = pipeline(
                    "text-generation",
                    model="google/flan-t5-small",
                    max_length=1024
                )
                logger.info("Text generation model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading text generation model: {e}")
                self.text_generator = None
            
            # Try loading document data in order of preference
            self.documents = []
            if os.path.exists(self.mapping_path):
                logger.info(f"Loading document mapping from {self.mapping_path}")
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from mapping")
            elif os.path.exists(self.processed_data_path):
                logger.info(f"Loading processed documents from {self.processed_data_path}")
                with open(self.processed_data_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} processed documents")
            else:
                logger.info(f"Loading original documents from {data_path}")
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    self.documents = self.data
                logger.info(f"Loaded {len(self.documents)} original documents")
                
            # Create a lookup dictionary for course codes
            self.course_lookup = {}
            for i, doc in enumerate(self.documents):
                course_code = doc.get("metadata", {}).get("course_code")
                if course_code:
                    if course_code not in self.course_lookup:
                        self.course_lookup[course_code] = []
                    self.course_lookup[course_code].append(i)
            
            logger.info(f"Created lookup dictionary for {len(self.course_lookup)} unique course codes")
                
        except Exception as e:
            logger.error(f"Error initializing RAGSearchTool: {e}")
            raise

    def _get_embedding(self, text):
        """Generate embeddings for the given text"""
        try:
            logger.info(f"Generating embedding for text of length {len(text)}")
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            logger.info(f"Embedding generated successfully.")
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros((384,), dtype='float32')
            
    def _extract_all_course_codes(self, text):
        """Extract all course codes from text using string operations.
        Returns a list of all found course codes.
        """
        course_codes = []
        text_upper = text.upper()
        words = text_upper.split()
        
        # Keywords that might indicate a comparison/list of courses
        comparison_indicators = ['OR', 'VS', 'VERSUS', 'COMPARE', 'BETWEEN', 'AND', 'WITH']
        
        has_comparison = any(indicator in words for indicator in comparison_indicators)
        
        for i, word in enumerate(words):
            if word == "DSAN" and i + 1 < len(words):
                # Check if next word is numeric or starts with a number
                next_word = words[i + 1].strip(":.,;?!")
                
                # Try to extract a numeric part
                if next_word.isdigit():
                    course_codes.append(f"DSAN {next_word}")
                # Handle case where number might be part of the word
                elif any(c.isdigit() for c in next_word):
                    # Extract just the numeric part
                    numeric_part = ''.join(c for c in next_word if c.isdigit())
                    if numeric_part:
                        course_codes.append(f"DSAN {numeric_part}")
        
        # Log what we found
        if course_codes:
            if len(course_codes) > 1:
                logger.info(f"Found multiple course codes: {course_codes}")
            else:
                logger.info(f"Found course code: {course_codes[0]}")
        else:
            logger.info("No course codes found in query")
            
        return course_codes, has_comparison

    def _extract_course_info(self, content):
        """Extract key information from course content"""
        info = {
            "title": "",
            "description": "",
            "credits": "",
            "schedule": "",
            "type": "Unknown",  # Core or Elective
        }
        
        # Extract title
        title_match = re.search(r'DSAN \d+:\s*([^\n]+)', content)
        if title_match:
            info["title"] = title_match.group(1).strip()
        
        # Extract if it's core or elective
        if "(Core)" in content:
            info["type"] = "Core"
        elif "(Elective)" in content:
            info["type"] = "Elective"
        
        # Extract description - everything after the title that's not a header
        lines = content.split("\n")
        description_lines = []
        capture = False
        
        for line in lines:
            if line.startswith("###") and "DSAN" in line:
                capture = True
                continue
            if capture and line.strip() and not line.startswith("#"):
                description_lines.append(line)
        
        if description_lines:
            info["description"] = " ".join(description_lines).strip()
        
        # Extract credits
        credits_match = re.search(r'(\d+)\s*credits', content, re.IGNORECASE)
        if credits_match:
            info["credits"] = f"{credits_match.group(1)} credits"
        
        # Extract schedule information
        schedule_patterns = [
            r'Offered in the ([^\.]+)',
            r'(Fall|Spring|Summer) semester',
        ]
        
        for pattern in schedule_patterns:
            schedule_match = re.search(pattern, content, re.IGNORECASE)
            if schedule_match:
                info["schedule"] = schedule_match.group(1).strip()
                break
        
        return info

    def _generate_conversational_response(self, query, results):
        """Generate a conversational response for a comparison query"""
        if not results or len(results) == 0:
            return None
        
        # Single course response
        if len(results) == 1:
            course_info = self._extract_course_info(results[0]["content"])
            course_code = results[0].get("course_code", "this course")
            
            response = f"# {course_code}: {course_info['title']}\n\n"
            response += f"This is a {course_info['type']} course "
            if course_info['credits']:
                response += f"worth {course_info['credits']} "
            if course_info['schedule']:
                response += f"offered in the {course_info['schedule']}.\n\n"
            else:
                response += ".\n\n"
                
            response += "## COURSE DESCRIPTION\n"
            if course_info['description']:
                response += f"{course_info['description']}\n\n"
            else:
                response += "Detailed description not available.\n\n"
                
            response += f"For more information about {course_code}, please check the [course catalog]({results[0]['url']})."
            
            return response
        
        # Comparison between courses
        else:
            course_infos = []
            for result in results:
                info = self._extract_course_info(result["content"])
                info["code"] = result.get("course_code", "Unknown")
                course_infos.append(info)
            
            # Generate a comparison
            response = f"# Comparing {' vs '.join([info['code'] for info in course_infos])}\n\n"
            response += "I'd be happy to help you compare these courses. Here's what you should know about each:\n\n"
            
            for info in course_infos:
                response += f"## {info['code']}: {info['title']}\n"
                response += f"- **Type**: {info['type']}\n"
                if info['credits']:
                    response += f"- **Credits**: {info['credits']}\n"
                if info['schedule']:
                    response += f"- **Schedule**: {info['schedule']}\n"
                response += f"- **Description**: {info['description'] if info['description'] else 'Not available'}\n\n"
            
            # Compare and provide recommendations
            response += "## Comparison\n"
            if all(info["type"] == "Core" for info in course_infos):
                response += "Both courses are core requirements in the DSAN program, which means you'll likely need to take both eventually.\n\n"
            elif any(info["type"] == "Core" for info in course_infos):
                core_courses = [info["code"] for info in course_infos if info["type"] == "Core"]
                elective_courses = [info["code"] for info in course_infos if info["type"] == "Elective"]
                response += f"{', '.join(core_courses)} {'is a' if len(core_courses) == 1 else 'are'} core requirement{'s' if len(core_courses) > 1 else ''}, while {', '.join(elective_courses)} {'is an' if len(elective_courses) == 1 else 'are'} elective{'s' if len(elective_courses) > 1 else ''}. You'll need to complete core requirements for your degree.\n\n"
            
            response += "## Recommendation\n"
            response += "Consider your interests and career goals:\n\n"
            
            for info in course_infos:
                interests = []
                if "programming" in info['description'].lower() or "python" in info['description'].lower():
                    interests.append("programming")
                if "analytics" in info['description'].lower() or "analysis" in info['description'].lower():
                    interests.append("data analysis")
                if "big data" in info['description'].lower() or "cloud" in info['description'].lower():
                    interests.append("cloud computing and big data")
                if "language" in info['description'].lower() or "nlp" in info['description'].lower():
                    interests.append("natural language processing")
                if "machine" in info['description'].lower() or "deep" in info['description'].lower():
                    interests.append("machine learning")
                
                if interests:
                    response += f"- Choose {info['code']} if you're interested in {', '.join(interests)}.\n"
                else:
                    response += f"- {info['code']}: Refer to the course description to see if it aligns with your interests.\n"
            
            return response
            
    def search(self, query, top_k=1):
        """Intelligent search that handles multiple course codes and comparisons"""
        try:
            logger.info(f"Performing search for query: '{query}' with top_k={top_k}")
            
            # Step 1: Check if query contains course codes and comparison indicators
            course_codes, has_comparison = self._extract_all_course_codes(query)
            
            # If we have course codes, retrieve information for each
            if course_codes:
                all_results = []
                
                # For each course code, find the best matching document
                for course_code in course_codes:
                    if course_code in self.course_lookup:
                        logger.info(f"Finding information for {course_code}")
                        
                        # Get the best document for this course code
                        best_doc = None
                        best_score = float('inf')
                        
                        for idx in self.course_lookup[course_code]:
                            if idx < len(self.documents):
                                doc = self.documents[idx]
                                
                                # Simple heuristic - prefer consolidated docs when available
                                doc_type = doc.get("metadata", {}).get("doc_type", "")
                                score = 0 if doc_type == "course_consolidated" else 1
                                
                                if score < best_score:
                                    best_doc = doc
                                    best_score = score
                        
                        # If we found a document, add it to results
                        if best_doc:
                            all_results.append({
                                "content": best_doc.get("markdown", ""),
                                "url": best_doc.get("metadata", {}).get("url", ""),
                                "title": best_doc.get("metadata", {}).get("title", "") or f"{course_code} Information",
                                "score": 0.0,  # Perfect match gets 0 distance
                                "is_exact_match": True,
                                "course_code": course_code
                            })
                
                # Generate a conversational response for comparison queries
                if has_comparison and len(all_results) > 1:
                    logger.info(f"Generating conversational comparison for {len(all_results)} courses")
                    
                    # Generate the comparison
                    conversational_response = self._generate_conversational_response(query, all_results)
                    
                    if conversational_response:
                        # Create a synthetic result that contains the comparison
                        comparison_result = {
                            "content": conversational_response,
                            "url": all_results[0]["url"],  # Use the URL from the first result
                            "title": f"Comparison: {' vs '.join([r.get('course_code', 'Course') for r in all_results])}",
                            "score": 0.0,
                            "is_exact_match": True,
                            "is_comparison": True
                        }
                        
                        # Return just the comparison result
                        return [comparison_result]
                    else:
                        # Fallback to returning individual results if comparison generation fails
                        logger.info(f"Returning information for comparison of {len(all_results)} courses (raw)")
                        return all_results
                
                # For non-comparison queries with course codes, generate a conversational response for one course
                elif all_results:
                    # Generate a conversational response for a single course
                    conversational_response = self._generate_conversational_response(query, [all_results[0]])
                    
                    if conversational_response:
                        # Create a synthetic result with the conversational response
                        enhanced_result = {
                            "content": conversational_response,
                            "url": all_results[0]["url"],
                            "title": all_results[0]["title"],
                            "score": 0.0,
                            "is_exact_match": True,
                            "course_code": all_results[0].get("course_code")
                        }
                        
                        # Return the enhanced result
                        return [enhanced_result]
                    else:
                        # Fallback to the original result
                        logger.info(f"Returning information for {all_results[0].get('course_code')}")
                        return [all_results[0]]
            
            # Fallback to semantic search if no course codes found or no matches
            logger.info(f"No direct course matches found, performing semantic search")
            
            # Perform semantic search with the embedding model
            query_embedding = self._get_embedding(query).astype('float32').reshape(1, -1)
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            
            # Use higher top_k for semantic search to ensure diversity
            semantic_top_k = 5 if top_k <= 1 else top_k
            
            distances, indices = self.index.search(query_embedding, semantic_top_k)
            logger.info(f"Distances: {distances}")
            logger.info(f"Indices: {indices}")
            
            semantic_results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    semantic_results.append({
                        "content": doc.get("markdown", ""),
                        "url": doc.get("metadata", {}).get("url", ""),
                        "title": doc.get("metadata", {}).get("title", ""),
                        "score": float(distances[0][i]),
                        "is_exact_match": False
                    })
            
            logger.info(f"Returning {len(semantic_results)} semantic search results")
            return semantic_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

# Initialize the RAG Search Tool
rag_search_tool = RAGSearchTool(INDEX_PATH, DATA_PATH)

# FastAPI App Initialization
app = FastAPI(
    title="Georgetown DSAN Program Information Agent",
    description="AI assistant for finding Georgetown University DSAN program information"
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# API Input Model
class QuestionRequest(BaseModel):
    question: str
    thread_id: int

# POST Endpoint
@app.post("/generate")
async def generate_route(request: QuestionRequest):
    try:
        logger.info(f"Received POST request with question: '{request.question}' and thread ID: {request.thread_id}")
        
        # Use our enhanced search method that handles multiple course codes
        results = rag_search_tool.search(request.question)
        logger.info(f"Results from RAG tool: {results}")
        
        outputs = []
        for result in results:
            outputs.append({
                "role": "response",
                "content": result["content"],
                "url": result["url"],
                "title": result["title"]
            })
        
        logger.info(f"Returning outputs: {outputs}")
        return {"result": outputs}
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Tool Definition
@tool
def get_course_info(query: str) -> dict:
    """Get comprehensive information about the DSAN program at Georgetown University."""
    try:
        logger.info(f"Retrieving course information for query: '{query}'")
        
        # Use our enhanced search method
        results = rag_search_tool.search(query)
        
        source_urls = []
        all_content = ""
        
        for result in results:
            if result.get("url"):
                source_urls.append(result["url"])
            if result.get("content"):
                all_content += result["content"] + "\n\n"
        
        logger.info(f"Retrieved course info successfully.")
        return {
            "query": query,
            "source_urls": list(set(source_urls)),
            "raw_search_results": results,
            "all_content": all_content[:10000]
        }
    except Exception as e:
        logger.error(f"Error in get_course_info: {e}")
        return {
            "query": query,
            "source_urls": [],
            "error": str(e)
        }

# Run the FastAPI App
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Georgetown DSAN Program Information Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)