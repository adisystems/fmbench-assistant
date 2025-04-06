import os
import json
import re
from typing import Dict, List, Any
from pathlib import Path

def parse_git_ingest(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a git ingest text file and convert it to a structured list of file information.
    
    Args:
        file_path: Path to the git ingest text file
        
    Returns:
        List of dictionaries containing file metadata and content
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # First, extract the directory structure if present
    files_data = []
    
    # Use regex to extract file blocks
    # Each file starts with "FILE:" pattern and ends with the start of next file or end of content
    file_pattern = re.compile(r"FILE:\s*(.*?)\s*\n(.*?)(?=\nFILE:|$)", re.DOTALL)
    
    for match in file_pattern.finditer(content):
        file_path = match.group(1).strip()
        file_content = match.group(2).strip()
        
        # Extract file metadata
        file_name = os.path.basename(file_path)
        directory = os.path.dirname(file_path)
        extension = os.path.splitext(file_name)[1].lstrip('.')
        
        file_data = {
            "filename": file_name,
            "path": file_path,
            "directory": directory,
            "extension": extension,
            "content": file_content
        }
        
        files_data.append(file_data)
    
    return files_data

def extract_directory_structure(file_path: str) -> str:
    """
    Extract directory structure section from the git ingest file
    
    Args:
        file_path: Path to the git ingest text file
        
    Returns:
        String containing directory structure
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find directory structure section
    dir_match = re.search(r"Directory structure:\s*\n(.*?)(?=\nFILE:)", content, re.DOTALL)
    
    if dir_match:
        return dir_match.group(1).strip()
    
    return ""

def convert_git_ingest_to_json(input_file: str, output_file: str) -> None:
    """
    Convert git ingest text file to JSON format
    
    Args:
        input_file: Path to the git ingest text file
        output_file: Path to save the JSON output
    """
    files_data = parse_git_ingest(input_file)
    directory_structure = extract_directory_structure(input_file)
    
    # Create final JSON structure
    output_data = {
        "metadata": {
            "total_files": len(files_data),
            "file_types": sorted(list(set(f["extension"] for f in files_data if f["extension"])))
        },
        "directory_structure": directory_structure,
        "files": files_data
    }
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Extracted information for {len(files_data)} files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert git ingest text file to JSON")
    parser.add_argument("input_file", help="Path to the git ingest text file")
    parser.add_argument("--output", "-o", default=None, 
                       help="Path to save the JSON output (default: input_file with .json extension)")
    
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}.json"
    
    convert_git_ingest_to_json(args.input_file, args.output)
