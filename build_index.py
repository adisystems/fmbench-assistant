import logging
import argparse
from pathlib import Path
from dsan_rag_setup import DSANRagSetup

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

def main():
    """Main function for creating and saving the vector index"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create and save a FAISS vector index for the DSAN RAG system")
    parser.add_argument("--data-file", type=str, default="data/documents_1.json", 
                        help="Path to the JSON data file containing documents")
    parser.add_argument("--vector-db-path", type=str, default="indexes/dsan_index", 
                        help="Path to save the FAISS vector database")
    parser.add_argument("--region", type=str, default="us-east-1", 
                        help="AWS region for Bedrock services")
    parser.add_argument("--embedding-model", type=str, default="amazon.titan-embed-text-v1", 
                        help="Amazon Bedrock embedding model ID")
    parser.add_argument("--bedrock-role-arn", type=str, 
                        default="arn:aws:iam::605134468121:role/BedrockCrossAccount2",
                        help="ARN of the IAM role to assume for Bedrock cross-account access")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting FAISS index creation process")
        logger.info(f"Data file: {args.data_file}")
        logger.info(f"Vector DB path: {args.vector_db_path}")
        
        # Create the RAG setup object
        rag_setup = DSANRagSetup(
            region=args.region,
            data_file_path=Path(args.data_file),
            embedding_model_id=args.embedding_model,
            vector_db_path=args.vector_db_path,
            bedrock_role_arn=args.bedrock_role_arn
        )
        
        # Create and save the index
        logger.info("Creating vector index - this may take some time depending on the document count")
        rag_setup.create_index()
        logger.info("🎉 Index creation completed successfully!")
        logger.info(f"Index saved to: {args.vector_db_path}")
        logger.info("You can now use this index for faster RAG queries")
    
    except Exception as e:
        logger.error(f"❌ Error creating index: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())