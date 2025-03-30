#!/usr/bin/env python3
import argparse
import boto3
import time
import sys
import subprocess
import os
import random

def wait_for_function_update_completion(lambda_client, function_name):
    """
    Wait for a Lambda function to complete any in-progress updates.
    
    Args:
        lambda_client: The boto3 Lambda client
        function_name: The name of the Lambda function
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            # Get the current state of the function
            response = lambda_client.get_function(FunctionName=function_name)
            last_update_status = response['Configuration'].get('LastUpdateStatus')
            
            # If LastUpdateStatus is not present or is "Successful", the update is complete
            if not last_update_status or last_update_status == 'Successful':
                print(f"Function update completed successfully")
                return True
            
            # If the update failed, report the error
            if last_update_status == 'Failed':
                failure_reason = response['Configuration'].get('LastUpdateStatusReason', 'Unknown error')
                print(f"Function update failed: {failure_reason}")
                return False
            
            # Still in progress, wait and retry
            print(f"Function update status: {last_update_status}. Waiting...")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error checking function status: {str(e)}")
            time.sleep(2)
    
    print(f"Timed out waiting for function update to complete")
    return False


def build_and_push_container():
    """
    Calls the build_and_push.sh script to build and push a Docker container to ECR.
    
    Returns:
        str: The ECR image URI if successful, None otherwise
    """
    print("=" * 80)
    print("No image URI provided. Building and pushing container to ECR...")
    print("=" * 80)
    
    try:
        # Ensure the script is executable
        script_path = "./build_and_push.sh"
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Script {script_path} not found. Please make sure it exists in the current directory.")
        
        # Execute the build_and_push.sh script
        result = subprocess.run(
            [script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Extract the ECR image URI from the script output
        # Assuming the script outputs the ECR URI as the last line or in a specific format
        output_lines = result.stdout.strip().split('\n')
        ecr_uri = output_lines[-1].strip()
        
        # Validate the URI (basic check)
        if not (ecr_uri.startswith("https://") or
                ".dkr.ecr." in ecr_uri or
                ".amazonaws.com/" in ecr_uri):
            print(f"Warning: The returned URI '{ecr_uri}' doesn't look like a valid ECR URI.")
            print("Full script output:")
            print(result.stdout)
            return None
        
        print(f"Successfully built and pushed container to: {ecr_uri}")
        return ecr_uri
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing build_and_push.sh: {e}")
        print(f"Script output: {e.stdout}")
        print(f"Script error: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error during build and push process: {str(e)}")
        return None

def deploy_lambda_container(ecr_image_uri, function_name, role_arn, region="us-east-1", memory_size=1024, timeout=90, create_url=True):
    """
    Deploy a container from ECR as a Lambda function with an optional function URL
    
    Args:
        ecr_image_uri (str): URI of the ECR image to deploy
        function_name (str): Name of the Lambda function
        role_arn (str): ARN of the Lambda execution role
        region (str): AWS region to deploy the Lambda function
        memory_size (int): Memory size in MB for the Lambda function
        timeout (int): Timeout in seconds for the Lambda function
        create_url (bool): Whether to create a function URL for direct HTTP access
    """
    print("=" * 80)
    print(f"Deploying container {ecr_image_uri} as Lambda function {function_name} in region {region}...")
    print("=" * 80)
    
    # Initialize the Lambda client with specified region
    lambda_client = boto3.client('lambda', region_name=region)
    
    try:
        # Check if function already exists
        try:
            lambda_client.get_function(FunctionName=function_name)
            function_exists = True
            print(f"Function {function_name} already exists. Updating...")
        except lambda_client.exceptions.ResourceNotFoundException:
            function_exists = False
            print(f"Function {function_name} does not exist. Creating new function...")
        
        # Create or update Lambda function
        if function_exists:
            # Update function code with retry logic
            max_code_retries = 5
            for attempt in range(max_code_retries):
                try:
                    response = lambda_client.update_function_code(
                        FunctionName=function_name,
                        ImageUri=ecr_image_uri,
                        Publish=True
                    )
                    print(f"Function code update initiated successfully")
                    break
                except lambda_client.exceptions.ResourceConflictException as e:
                    if attempt < max_code_retries - 1:
                        wait_time = (2 ** attempt) + (random.random() * 0.5)  # Exponential backoff with jitter
                        print(f"Update already in progress. Waiting {wait_time:.2f} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            # Wait for function code update to complete before updating configuration
            print("Waiting for function code update to complete...")
            wait_for_function_update_completion(lambda_client, function_name)
            
            # Update configuration with retry logic
            max_config_retries = 5
            for attempt in range(max_config_retries):
                try:
                    lambda_client.update_function_configuration(
                        FunctionName=function_name,
                        Timeout=timeout,
                        MemorySize=memory_size
                    )
                    print(f"Function configuration updated successfully")
                    break
                except lambda_client.exceptions.ResourceConflictException as e:
                    if attempt < max_config_retries - 1:
                        wait_time = (2 ** attempt) + (random.random() * 0.5)  # Exponential backoff with jitter
                        print(f"Update already in progress. Waiting {wait_time:.2f} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        raise e
        else:
            # Create new function
            response = lambda_client.create_function(
                FunctionName=function_name,
                PackageType='Image',
                Code={
                    'ImageUri': ecr_image_uri
                },
                Role=role_arn,
                Timeout=timeout,
                MemorySize=memory_size
            )
        
        # Wait for function to be active
        print("Waiting for Lambda function to be ready...")
        function_state = ""
        max_attempts = 10
        attempts = 0
        
        while function_state != "Active" and attempts < max_attempts:
            time.sleep(5)
            attempts += 1
            function_info = lambda_client.get_function(FunctionName=function_name)
            function_state = function_info['Configuration']['State']
            print(f"Current state: {function_state}")
        
        if function_state == "Active":
            print(f"Successfully deployed Lambda function: {function_name}")
            function_arn = function_info['Configuration']['FunctionArn']
            print(f"Function ARN: {function_arn}")
            
            # Create or update function URL configuration
            print("Setting up Lambda function URL...")
            try:
                # Check if function URL already exists
                try:
                    url_config = lambda_client.get_function_url_config(FunctionName=function_name)
                    print(f"Function URL already exists: {url_config['FunctionUrl']}")
                    
                    # Update function URL config (if needed)
                    lambda_client.update_function_url_config(
                        FunctionName=function_name,
                        AuthType='NONE',  # Public access, no IAM authentication
                        Cors={
                            'AllowOrigins': ['*'],
                            'AllowMethods': ['*'],  # Use wildcard instead of listing specific methods
                            'AllowHeaders': ['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key', 'X-Amz-Security-Token', 'X-Amz-User-Agent', 'Access-Control-Allow-Origin'],
                            'ExposeHeaders': ['*'],
                            'MaxAge': 86400  # 24 hours
                        }
                    )
                    print("Updated function URL configuration with CORS settings")
                    
                    # Ensure permissions are set correctly for existing URL
                    try:
                        # Check if permission already exists
                        try:
                            policy = lambda_client.get_policy(FunctionName=function_name)
                            if '"Principal":"*"' in policy['Policy'] and 'lambda:InvokeFunctionUrl' in policy['Policy']:
                                print("Public access permission already exists")
                            else:
                                # Add missing permission
                                lambda_client.add_permission(
                                    FunctionName=function_name,
                                    StatementId='FunctionURLAllowPublicAccess',
                                    Action='lambda:InvokeFunctionUrl',
                                    Principal='*',
                                    FunctionUrlAuthType='NONE'
                                )
                                print("Added missing permission for public access to function URL")
                        except lambda_client.exceptions.ResourceNotFoundException:
                            # No policy exists yet, add one
                            lambda_client.add_permission(
                                FunctionName=function_name,
                                StatementId='FunctionURLAllowPublicAccess',
                                Action='lambda:InvokeFunctionUrl',
                                Principal='*',
                                FunctionUrlAuthType='NONE'
                            )
                            print("Added permission for public access to function URL")
                    except Exception as e:
                        print(f"Warning: Error checking/setting permissions: {str(e)}")
                        print("You may need to manually set permissions for browser access.")
                except lambda_client.exceptions.ResourceNotFoundException:
                    # Create new function URL
                    url_config = lambda_client.create_function_url_config(
                        FunctionName=function_name,
                        AuthType='NONE',  # Public access, no IAM authentication
                        Cors={
                            'AllowOrigins': ['*'],
                            'AllowMethods': ['*'],  # Use wildcard instead of listing specific methods
                            'AllowHeaders': ['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key', 'X-Amz-Security-Token', 'X-Amz-User-Agent'],
                            'ExposeHeaders': ['*'],
                            'MaxAge': 86400  # 24 hours
                        }
                    )
                    print(f"Created new function URL: {url_config['FunctionUrl']}")
                    
                    # Add permission for public access from any source (including browsers)
                    try:
                        lambda_client.add_permission(
                            FunctionName=function_name,
                            StatementId='FunctionURLAllowPublicAccess',
                            Action='lambda:InvokeFunctionUrl',
                            Principal='*',
                            FunctionUrlAuthType='NONE'
                        )
                        print("Added permission for public access to function URL")
                        
                        # Verify the permission was added correctly
                        policy = lambda_client.get_policy(FunctionName=function_name)
                        print("Function policy successfully configured for public access")
                    except lambda_client.exceptions.ResourceConflictException:
                        # Permission might already exist, which is fine
                        print("Public access permission already exists for this function")
                    except Exception as e:
                        print(f"Warning: Error setting permissions: {str(e)}")
                        print("The function URL may not be accessible from browsers. You may need to manually set permissions.")
                
                return True
                
            except Exception as e:
                print(f"Error setting up function URL: {str(e)}")
                print("Function was deployed successfully, but function URL setup failed")
                return True
        else:
            print(f"Function deployment did not reach Active state in time. Last state: {function_state}")
            return False
            
    except Exception as e:
        print(f"Error deploying Lambda function: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Deploy a container from ECR as a Lambda function')
    parser.add_argument('--image-uri', required=False, help='ECR image URI to deploy (if not provided, will build and push container)')
    parser.add_argument('--function-name', required=True, help='Name for the Lambda function')
    parser.add_argument('--role-arn', required=True, help='ARN of the Lambda execution role')
    parser.add_argument('--region', default='us-east-1', help='AWS region to deploy the Lambda function (default: us-east-1)')
    parser.add_argument('--memory', type=int, default=1024, help='Memory size in MB (default: 1024)')
    parser.add_argument('--timeout', type=int, default=90, help='Timeout in seconds (default: 90)')
    parser.add_argument('--no-url', action='store_true', help='Do not create a function URL (by default, a function URL is created)')
    
    args = parser.parse_args()
    
    # Determine if we need to build and push a container or use the provided URI
    ecr_image_uri = args.image_uri
    if not ecr_image_uri:
        ecr_image_uri = build_and_push_container()
        if not ecr_image_uri:
            print("Failed to build and push container. Exiting.")
            sys.exit(1)
    else:
        print("=" * 80)
        print(f"Using provided image URI: {ecr_image_uri}")
        print("=" * 80)
    
    # Deploy the Lambda function with the image URI
    success = deploy_lambda_container(
        ecr_image_uri, 
        args.function_name, 
        args.role_arn,
        args.region,
        args.memory,
        args.timeout,
        not args.no_url  # Create URL by default unless --no-url flag is provided
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()