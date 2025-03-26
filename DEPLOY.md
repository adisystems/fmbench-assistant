# Deploying to AWS Lambda

This guide explains how to deploy the Solar Panel Savings Calculator API to AWS Lambda using container images.

## Prerequisites

- AWS CLI installed and configured with appropriate credentials
- Docker installed on your development machine
- An AWS ECR repository to store your container images
- IAM permissions to push to ECR and deploy to Lambda

## Deployment Steps

### 1. Build the Docker Image

The project includes a Dockerfile that sets up the environment for the FastAPI application. Build the image using:

```bash
docker build -t solar-calculator-api .
```

### 2. Push to Amazon ECR

1. Create an ECR repository (if not already created):
```bash
aws ecr create-repository --repository-name solar-calculator-api
```

2. Authenticate Docker to your ECR registry:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
```

3. Tag and push the image:
```bash
docker tag solar-calculator-api:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/solar-calculator-api:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/solar-calculator-api:latest
```

### 3. Create Lambda Function

1. Create a new Lambda function using the container image:
   - Go to AWS Lambda console
   - Click "Create function"
   - Choose "Container image"
   - Select the image you pushed to ECR
   - Configure memory (recommended: 2048 MB)
   - Configure timeout (recommended: 30 seconds)

2. Configure environment variables in Lambda:
   - TAVILY_API_KEY: Your Tavily API key
   - Any other environment variables needed by your application

### 4. Configure Function URL

1. Enable function URL in the Lambda console:
   - Go to Configuration → Function URL
   - Click "Create function URL"
   - Auth type: NONE (for public access)
   - Configure CORS if needed

### 5. IAM Permissions

Ensure your Lambda function has these permissions:
- AWSLambdaBasicExecutionRole (for CloudWatch Logs)
- Permissions to access Bedrock (through IAM policy)

Example Bedrock policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "*"
        }
    ]
}
```

### 6. Testing the Deployment

Test the deployed API using curl:

```bash
curl -X POST https://[your-function-url].lambda-url.us-east-1.on.aws/ \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What would be my savings with a monthly electricity cost of 200?",
    "thread_id": 123
  }'
```

## Automated Deployment

The repository includes a `build_and_push.sh` script that automates the build and push process:

```bash
./build_and_push.sh
```

## Troubleshooting

1. Check CloudWatch Logs for errors
2. Verify environment variables are set correctly
3. Ensure IAM roles and permissions are properly configured
4. Check Lambda timeout and memory settings if requests fail

## Monitoring

- Monitor Lambda metrics in CloudWatch
- Set up CloudWatch Alarms for errors and timeouts
- Track API usage through CloudWatch Logs

## Cost Optimization

- Configure memory based on actual usage patterns
- Monitor and adjust timeout settings
- Use provisioned concurrency if needed for consistent performance