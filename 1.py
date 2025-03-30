import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import random
from requests.exceptions import RequestException

def search_with_backoff(query, max_results=5, initial_sleep=0.5, max_retries=5, backoff_factor=2):
    """
    Performs Google search with exponential backoff for handling rate limits.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        initial_sleep: Initial sleep time between requests in seconds
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplicative factor for backoff
        
    Returns:
        List of valid search result URLs
    """
    results = []
    current_sleep = initial_sleep
    retry_count = 0
    
    # Create a generator for search results
    search_generator = search(
        query, 
        advanced=True,
        sleep_interval=current_sleep,
        num_results=max_results * 2  # Request more to filter out bad results
    )
    
    while len(results) < max_results and retry_count < max_retries:
        try:
            # Try to get the next result
            result = next(search_generator, None)
            
            # If no more results or reached limit, break
            if result is None:
                break
                
            # Check if the result is a valid URL (not relative path)
            if result.startswith('http'):
                # Validate the URL by making a HEAD request
                try:
                    response = requests.head(
                        result, 
                        timeout=5,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    )
                    
                    if response.status_code == 200:
                        results.append(result)
                        print(f"Found valid result: {result}")
                    elif response.status_code == 429:
                        raise RequestException("Rate limited with 429 status code")
                except RequestException as e:
                    print(f"Error validating URL {result}: {str(e)}")
                    # If rate limited, apply backoff
                    if "429" in str(e):
                        raise RequestException("Rate limited with 429 status code")
            else:
                print(f"Skipping invalid URL: {result}")
                
        except RequestException as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                retry_count += 1
                # Apply exponential backoff with jitter
                current_sleep = current_sleep * backoff_factor + random.uniform(0, 1)
                print(f"Rate limited. Retry {retry_count}/{max_retries}. Waiting {current_sleep:.2f} seconds...")
                time.sleep(current_sleep)
                
                # Recreate the generator with new sleep interval
                search_generator = search(
                    query, 
                    advanced=True,
                    sleep_interval=current_sleep,
                    num_results=max_results * 2
                )
            else:
                print(f"Error during search: {str(e)}")
                retry_count += 1
                time.sleep(current_sleep)
        
        # Small sleep between successful requests to be polite
        time.sleep(initial_sleep)
    
    return results

# Main code execution
if __name__ == "__main__":
    preamble = "Unless specified otherwise this is a query about courses offered by Georgetown University. "
    query = f"{preamble} what is dsan6725?"
    print(f"Searching for: {query}")
    
    try:
        valid_results = search_with_backoff(
            query,
            max_results=5,
            initial_sleep=1.0,
            max_retries=5,
            backoff_factor=2
        )
        
        print("\nSearch Results:")
        for i, result in enumerate(valid_results, 1):
            print(f"{i}. {result}")
            
        if not valid_results:
            print("No valid results found. Try again later or adjust parameters.")
            
    except Exception as e:
        print(f"Search failed: {str(e)}")
