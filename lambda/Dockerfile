FROM public.ecr.aws/lambda/python:3.11

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code and data
COPY lambda.py ${LAMBDA_TASK_ROOT}
COPY __init__.py ${LAMBDA_TASK_ROOT}
COPY ../data/documents_1.json ${LAMBDA_TASK_ROOT}/data/

# Set the CMD to your handler
CMD [ "lambda.handler" ]