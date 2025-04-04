FROM public.ecr.aws/lambda/python:3.11

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -U boto3 botocore

# Copy function code and data
COPY dsan_rag_setup.py ${LAMBDA_TASK_ROOT}
COPY guardrails.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
COPY app/server.py ${LAMBDA_TASK_ROOT}/lambda.py
COPY app/__init__.py ${LAMBDA_TASK_ROOT}
COPY indexes ${LAMBDA_TASK_ROOT}/indexes

# Set the CMD to your handler
CMD [ "lambda.handler" ]