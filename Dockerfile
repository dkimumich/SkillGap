# AWS's official Lambda base image for Python. It already contains the
# Lambda runtime, so the container knows how to receive events from AWS.
FROM public.ecr.aws/lambda/python:3.12

# Copy the project in and install it (plus scikit-learn, FastAPI, Mangum).
COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
RUN pip install --no-cache-dir ${LAMBDA_TASK_ROOT}

# The function Lambda calls for every request: module path + variable name.
CMD ["skillgap.lambda_handler.handler"]
