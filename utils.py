import boto3
import logging
from typing import Optional
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session

# Assuming logger is defined elsewhere or we can add it here
logger = logging.getLogger(__name__)

def create_bedrock_client(bedrock_role_arn: Optional[str], service: str, region: str):
    """Create a Bedrock client, optionally with cross-account role assumption"""
    config = Config(
        retries={
            'max_attempts': 10,
            'mode': 'adaptive'
        }
    )
    
    # If a role ARN is provided, use cross-account access
    if bedrock_role_arn:
        logger.info(f"Initializing Bedrock client with cross-account role: {bedrock_role_arn}")
        
        def get_credentials():
            sts_client = boto3.client('sts')
            assumed_role = sts_client.assume_role(
                RoleArn=bedrock_role_arn,
                RoleSessionName='bedrock-cross-account-session',
                # Don't set DurationSeconds when role chaining
            )
            return {
                'access_key': assumed_role['Credentials']['AccessKeyId'],
                'secret_key': assumed_role['Credentials']['SecretAccessKey'],
                'token': assumed_role['Credentials']['SessionToken'],
                'expiry_time': assumed_role['Credentials']['Expiration'].isoformat()
            }

        session = get_session()
        refresh_creds = RefreshableCredentials.create_from_metadata(
            metadata=get_credentials(),
            refresh_using=get_credentials,
            method='sts-assume-role'
        )

        # Create a new session with refreshable credentials
        session._credentials = refresh_creds
        boto3_session = boto3.Session(botocore_session=session)
        
        return boto3_session.client(service, region_name=region, config=config)
    else:
        logger.info(f"Initializing Bedrock client for region: {region}")
        return boto3.client(service, region_name=region, config=config)