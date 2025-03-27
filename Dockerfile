FROM amazon/aws-lambda-python:3.11

# Install uv
RUN pip install --upgrade pip
RUN pip install uv

# Copy project files
COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
COPY README.md ${LAMBDA_TASK_ROOT}/
COPY LICENSE ${LAMBDA_TASK_ROOT}/

# Install dependencies using uv
RUN cd ${LAMBDA_TASK_ROOT} && uv venv && uv pip install -e .
RUN uv pip install -U boto3 botocore

# Copy function code
COPY lambda/ ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "index.handler" ]