"""AWS Lambda entry point.

API Gateway sends each HTTP request to Lambda as a JSON "event". Mangum
translates that event into a normal web request for the FastAPI app, then
translates FastAPI's response back into the format API Gateway expects.

Lambda is configured to call ``skillgap.lambda_handler.handler``.
"""

from mangum import Mangum

from skillgap.api import app

handler = Mangum(app, lifespan="off")
