import os
import sys
import json
import logging
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("indexer_debug.log")  # Also log to a file
    ]
)
logger = logging.getLogger(__name__)

# Force CPU usage globally
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

# Configuration with absolute paths for clarity
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.debug(f"Base directory: {BASE_DIR}")

DATA_PATH = os.path.join(BASE_DIR, "data", "documents_1.json")
INDEX_PATH = os.path.join(BASE_DIR, "data", "dsan_faiss.index")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_documents.json")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "documents_mapping.json")

logger.debug(f"DATA_PATH: {DATA_PATH}")
logger.debug(f"INDEX_PATH: {INDEX_PATH}")
logger.debug(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
logger.debug(f"MAPPING_PATH: {MAPPING_PATH}")

# Ensure data directory exists
data_dir = os.path.dirname(DATA_PATH)
if not os.path.exists(data_dir):
    try:
        os.makedirs(data_dir)
        logger.debug(f"Created data directory: {data_dir}")
    except Exception as e:
        logger.error(f"Failed to create data directory: {e}")

class DSANIndexer:
    def __init__(self, json_path, index_path, processed_data_path, mapping_path):
        """Initialize the DSAN indexer with model and tokenizer"""
        self.json_path = json_path
        self.index_path = index_path
        self.processed_data_path = processed_data_path
        self.mapping_path = mapping_path
        self.documents = []
        self.course_specific_docs = []

        # Check file existence
        logger.debug(f"Checking if json_path exists: {os.path.exists(json_path)}")
        if not os.path.exists(json_path):
            logger.error(f"Input JSON file does not exist: {json_path}")
            raise FileNotFoundError(f"Input JSON file does not exist: {json_path}")

        # Load the model and tokenizer
        try:
            logger.info("Loading tokenizer and model for embedding generation")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def _get_embedding(self, text):
        """Generate embeddings from text using the model."""
        try:
            logger.info(f"Generating embedding for text of length {len(text)}")
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            logger.info(f"Embedding generated successfully with shape {embeddings.shape}")
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros((384,), dtype='float32')
    
    def _extract_course_code(self, text):
        """Extract course code from text using string operations."""
        words = text.upper().split()
        for i, word in enumerate(words):
            if word == "DSAN" and i + 1 < len(words):
                # Check if next word is numeric or starts with a number
                next_word = words[i + 1].strip(":")
                if next_word.isdigit():
                    return f"DSAN {next_word}"
                # Handle case where number might be part of the word
                elif any(c.isdigit() for c in next_word):
                    # Extract just the numeric part
                    numeric_part = ''.join(c for c in next_word if c.isdigit())
                    if numeric_part:
                        return f"DSAN {numeric_part}"
        return None

    def _process_course_section(self, content, metadata):
        """Process a section of content to identify and extract course information."""
        course_sections = {}
        current_section = None
        section_content = ""
        lines = content.splitlines()
        
        # Debug line counts
        logger.debug(f"Processing content with {len(lines)} lines")
        
        # First pass: identify potential course headings and sections
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            course_code = self._extract_course_code(line)
            
            # If line contains DSAN and looks like a course header
            if course_code and (":" in line or "-" in line):
                logger.debug(f"Found course header: {line} with code {course_code}")
                # Save previous section if it exists
                if current_section and section_content:
                    course_sections[current_section] = section_content
                    logger.debug(f"Saved section for {current_section} with {len(section_content)} chars")
                
                # Start new section
                current_section = course_code
                section_content = line + "\n"
            elif current_section:
                # Continue adding to current section
                section_content += line + "\n"
                
        # Save the last section
        if current_section and section_content:
            course_sections[current_section] = section_content
            logger.debug(f"Saved final section for {current_section} with {len(section_content)} chars")
            
        # Second pass: for content not classified as a course, check if it mentions courses
        if not course_sections:
            # Check if content mentions any course codes
            for line in lines:
                course_code = self._extract_course_code(line)
                if course_code:
                    logger.debug(f"Found mention of {course_code} in non-header content")
                    # Create a general entry for this content mentioning the course
                    if course_code not in course_sections:
                        course_sections[course_code] = content
                        logger.debug(f"Created general entry for {course_code}")
        
        logger.debug(f"Extracted {len(course_sections)} course sections")
        return course_sections

    def _create_enhanced_documents(self, data):
        """Create enhanced document structure with better course metadata."""
        enhanced_docs = []
        course_docs = {}
        
        logger.info(f"Creating enhanced documents from {len(data)} original docs")
        
        # First process each document to identify course-specific content
        for item in data:
            if "markdown" not in item or not item["markdown"]:
                continue
                
            content = item["markdown"]
            metadata = item.get("metadata", {})
            
            # Process the document to extract course sections
            course_sections = self._process_course_section(content, metadata)
            
            # If we found course sections, create specific documents for each
            if course_sections:
                for course_code, course_content in course_sections.items():
                    if course_code not in course_docs:
                        course_docs[course_code] = []
                    
                    course_docs[course_code].append({
                        "markdown": course_content,
                        "metadata": {
                            **metadata,
                            "course_code": course_code,
                            "doc_type": "course_specific"
                        }
                    })
                    logger.debug(f"Created specific doc for {course_code}")
            
            # Always keep the original document
            enhanced_docs.append(item)
        
        # Now create consolidated course documents
        for course_code, docs in course_docs.items():
            # Combine all content for this course
            combined_content = f"# {course_code}\n\n"
            urls = set()
            titles = set()
            
            for doc in docs:
                combined_content += doc["markdown"] + "\n\n"
                if "url" in doc["metadata"]:
                    urls.add(doc["metadata"]["url"])
                if "title" in doc["metadata"]:
                    titles.add(doc["metadata"]["title"])
            
            # Create a consolidated document
            enhanced_docs.append({
                "markdown": combined_content,
                "metadata": {
                    "course_code": course_code,
                    "doc_type": "course_consolidated",
                    "url": list(urls)[0] if urls else "",
                    "title": f"{course_code} Course Information" if not titles else list(titles)[0]
                }
            })
            logger.debug(f"Created consolidated doc for {course_code}")
        
        logger.info(f"Created a total of {len(enhanced_docs)} enhanced documents")
        return enhanced_docs

    def build_index(self):
        """Build and save the FAISS index with improved course structure."""
        try:
            logger.info(f"Loading data from {self.json_path}")
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} documents from {self.json_path}")
            
            # Process data to create enhanced document structure
            processed_data = self._create_enhanced_documents(data)
            
            # Save processed data for later use in search
            try:
                logger.info(f"Saving processed data to {self.processed_data_path}")
                with open(self.processed_data_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved processed data successfully")
            except Exception as e:
                logger.error(f"Error saving processed data: {e}")
                # Try an alternative location if the original fails
                alt_path = os.path.join(os.getcwd(), "processed_documents.json")
                logger.info(f"Trying alternative location: {alt_path}")
                with open(alt_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved processed data to alternative location: {alt_path}")
            
            embeddings = []
            self.documents = []
            
            # Generate embeddings for all documents
            for item in processed_data:
                if "markdown" in item and item["markdown"]:
                    try:
                        # For course-specific documents, create high-quality embedding
                        if item.get("metadata", {}).get("doc_type") in ["course_specific", "course_consolidated"]:
                            # Generate embedding for the whole document
                            embedding = self._get_embedding(item["markdown"])
                            embeddings.append(embedding)
                            self.documents.append(item)
                            
                            # Also create embeddings for queries like "tell me about DSAN XXXX"
                            course_code = item.get("metadata", {}).get("course_code")
                            if course_code:
                                query_text = f"tell me about {course_code}"
                                query_embedding = self._get_embedding(query_text)
                                embeddings.append(query_embedding)
                                # Duplicate the document for the query embedding
                                self.documents.append(item)
                        else:
                            # For general documents, keep the original approach
                            embedding = self._get_embedding(item["markdown"])
                            embeddings.append(embedding)
                            self.documents.append(item)
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
                        continue

            # Convert to numpy array
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize if we have embeddings
            if len(embeddings) > 0:
                faiss.normalize_L2(embeddings)
                logger.info(f"Total embeddings generated: {len(embeddings)}")
            else:
                logger.error("No embeddings generated. Exiting index creation.")
                return

            # Build the FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Save the index
            try:
                logger.info(f"Saving FAISS index to {self.index_path}")
                faiss.write_index(index, self.index_path)
                logger.info(f"FAISS index saved successfully")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")
                # Try an alternative location
                alt_path = os.path.join(os.getcwd(), "dsan_faiss.index")
                logger.info(f"Trying alternative location: {alt_path}")
                faiss.write_index(index, alt_path)
                logger.info(f"Saved FAISS index to alternative location: {alt_path}")
            
            logger.info(f"Number of documents indexed: {len(self.documents)}")
            
            # Save the documents mapping
            try:
                logger.info(f"Saving documents mapping to {self.mapping_path}")
                with open(self.mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(self.documents, f, ensure_ascii=False, indent=2)
                logger.info(f"Documents mapping saved successfully")
            except Exception as e:
                logger.error(f"Error saving documents mapping: {e}")
                # Try an alternative location
                alt_path = os.path.join(os.getcwd(), "documents_mapping.json")
                logger.info(f"Trying alternative location: {alt_path}")
                with open(alt_path, 'w', encoding='utf-8') as f:
                    json.dump(self.documents, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved documents mapping to alternative location: {alt_path}")
            
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        # Print working directory and python environment to help diagnose issues
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Python executable: {sys.executable}")
        
        # Check file permissions in data directory
        data_dir = os.path.dirname(DATA_PATH)
        logger.debug(f"Checking permissions on data directory: {data_dir}")
        if os.path.exists(data_dir):
            try:
                test_file = os.path.join(data_dir, "test_write.txt")
                with open(test_file, 'w') as f:
                    f.write("Test write")
                os.remove(test_file)
                logger.debug("Successfully wrote test file to data directory")
            except Exception as e:
                logger.error(f"Cannot write to data directory: {e}")
                
        # Create indexer and build index
        indexer = DSANIndexer(DATA_PATH, INDEX_PATH, PROCESSED_DATA_PATH, MAPPING_PATH)
        indexer.build_index()
        
        logger.info("Indexing completed successfully")
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)