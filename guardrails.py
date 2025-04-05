import boto3
import logging
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session


class GuardrailTopicExample(BaseModel):
    name: str
    definition: str
    examples: List[str]
    type: str


class GuardrailFilter(BaseModel):
    type: str
    inputStrength: str
    outputStrength: str


class GuardrailConfig(BaseModel):
    name: str = "fmbench-guardrails"
    description: str = Field(
        default="Ensures the chatbot provides accurate information about the AWS Foundation Model Benchmarking Tool (FMBench) while maintaining appropriate security and usage boundaries."
    )
    topic_policies: List[GuardrailTopicExample] = Field(
        default=[
            GuardrailTopicExample(
                name="Model Details",
                definition="Providing specific internal implementation details about foundation models being benchmarked.",
                examples=[
                    "How is model X implemented internally?",
                    "What's the architecture of this foundation model?",
                    "Share the model training details.",
                    "What datasets were used to train this model?",
                    "Explain the model's neural network structure.",
                ],
                type="DENY",
            ),
            GuardrailTopicExample(
                name="Cost Information",
                definition="Providing specific pricing, billing, or cost information about using foundation models or AWS services.",
                examples=[
                    "How much does it cost to use this model?",
                    "What's the AWS billing rate?",
                    "How many tokens can I use for free?",
                    "What's the pricing structure?",
                    "How much will my benchmarking cost?",
                ],
                type="DENY",
            ),
        ]
    )
    content_filters: List[GuardrailFilter] = Field(
        default=[
            GuardrailFilter(type="SEXUAL", inputStrength="HIGH", outputStrength="HIGH"),
            GuardrailFilter(type="VIOLENCE", inputStrength="HIGH", outputStrength="HIGH"),
            GuardrailFilter(type="HATE", inputStrength="HIGH", outputStrength="HIGH"),
            GuardrailFilter(type="INSULTS", inputStrength="HIGH", outputStrength="HIGH"),
            GuardrailFilter(type="MISCONDUCT", inputStrength="HIGH", outputStrength="HIGH"),
            GuardrailFilter(type="PROMPT_ATTACK", inputStrength="HIGH", outputStrength="NONE"),
        ]
    )
    blocked_input_messaging: str = Field(
        default=(
            "I apologize, but I cannot provide information about sensitive implementation details, security configurations, "
            "or specific performance claims without verified benchmarking data. For more information about FMBench, "
            "please refer to the official AWS documentation or contact the AWS support team."
        )
    )
    blocked_outputs_messaging: str = Field(
        default=(
            "I cannot provide that information as it may involve sensitive implementation details, security configurations, "
            "or unverified performance claims. For accurate information about FMBench and AWS services, please refer to "
            "the official AWS documentation or contact AWS support."
        )
    )


class BedrockGuardrailManager(BaseModel):
    region: str
    bedrock_role_arn: Optional[str] = None
    logger: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Set up logging if not provided
        if self.logger is None:
            self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Create and configure a logger"""
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
        
        return logger

    def _create_bedrock_client(self):
        """Create a Bedrock client, optionally with cross-account role assumption"""
        config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        
        # If a role ARN is provided, use cross-account access
        if self.bedrock_role_arn:
            self.logger.info(f"Initializing Bedrock client with cross-account role: {self.bedrock_role_arn}")
            
            def get_credentials():
                sts_client = boto3.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.bedrock_role_arn,
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
            
            return boto3_session.client("bedrock", region_name=self.region, config=config)
        else:
            self.logger.info(f"Initializing Bedrock client for region: {self.region}")
            return boto3.client("bedrock", region_name=self.region, config=config)

    def get_or_create_guardrail(self, guardrail_config: Optional[GuardrailConfig] = None) -> tuple[str, str]:
        """
        Get or create a Bedrock guardrail
        
        Args:
            guardrail_config: Optional configuration for the guardrail. If not provided, default values will be used.
            
        Returns:
            Tuple with guardrail ID and version
        """
        if guardrail_config is None:
            guardrail_config = GuardrailConfig()
            
        bedrock_client = self._create_bedrock_client()
        
        try:
            # First, check if a guardrail with this name already exists
            existing_guardrails = bedrock_client.list_guardrails()['guardrails']
            self.logger.info(f"Existing guardrails: {existing_guardrails}")
            for guardrail in existing_guardrails:
                if guardrail['name'] == guardrail_config.name:
                    self.logger.info(f"Guardrail already exists: {guardrail}")
                    return guardrail['id'], guardrail['version']

            # If not found, create it
            response = bedrock_client.create_guardrail(
                name=guardrail_config.name,
                description=guardrail_config.description,
                topicPolicyConfig={
                    'topicsConfig': [
                        {
                            'name': topic.name,
                            'definition': topic.definition,
                            'examples': topic.examples,
                            'type': topic.type
                        } for topic in guardrail_config.topic_policies
                    ]
                },
                contentPolicyConfig={
                    'filtersConfig': [
                        {
                            'type': filter.type,
                            'inputStrength': filter.inputStrength,
                            'outputStrength': filter.outputStrength
                        } for filter in guardrail_config.content_filters
                    ]
                },
                blockedInputMessaging=guardrail_config.blocked_input_messaging,
                blockedOutputsMessaging=guardrail_config.blocked_outputs_messaging,
            )
            self.logger.info(f"Guardrail created successfully: {response}")
            return response['guardrailId'], response['version']

        except Exception as e:
            self.logger.error(f"Failed to get or create guardrail: {str(e)}")
            raise