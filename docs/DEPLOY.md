# Deploying SkillGap to AWS

Architecture: **API Gateway (HTTP API) → Lambda (container image from ECR) → FastAPI via Mangum**

Commands are for **PowerShell on Windows**, run from the repo root. Region: `us-east-1`.

---

## Part A: One-time setup

1. **AWS account.** Sign up at aws.amazon.com. Turn on MFA for the root user (Account menu → Security credentials).
2. **Budget alert.** Billing → Budgets → Create budget → "Zero spend budget" or a $5 monthly budget with an email alert. This project should cost little to nothing, but the alert protects you from surprises.
3. **Admin user for the CLI** (don't use root day to day). IAM → Users → Create user `skillgap-admin` → Attach policy `AdministratorAccess`. Open the user → Security credentials → Create access key → "Command Line Interface". Save the two keys.
4. **Install the AWS CLI v2** (Windows MSI from AWS docs), then:
   ```powershell
   aws configure
   # Access key, secret key, region: us-east-1, output: json
   aws sts get-caller-identity   # should print your account ID
   ```
5. **Install Docker Desktop** (it uses WSL 2; let the installer set that up). Start it, then check `docker --version`.

---

## Part B: Deploy (first time)

Save your account ID and region in variables for this PowerShell session:
```powershell
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$REGION  = "us-east-1"
$REGISTRY = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$IMAGE    = "$REGISTRY/skillgap:latest"
```

### 1. Create the image repository (ECR)
```powershell
aws ecr create-repository --repository-name skillgap --region $REGION
```

### 2. Build and push the Docker image
```powershell
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker build --platform linux/amd64 --provenance=false -t $IMAGE .
docker push $IMAGE
```
- `--platform linux/amd64` builds for the CPU type Lambda will use.
- `--provenance=false` stops Docker from adding extra metadata that Lambda rejects
  ("image manifest media type not supported").

### 3. Create the Lambda execution role
```powershell
aws iam create-role --role-name skillgap-lambda-role `
  --assume-role-policy-document file://infra/lambda-trust-policy.json
aws iam attach-role-policy --role-name skillgap-lambda-role `
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```
The trust policy lets Lambda use the role; the attached policy lets it write logs to CloudWatch.
Wait ~10 seconds before the next step so the new role is ready.

### 4. Create the Lambda function
```powershell
aws lambda create-function --function-name skillgap `
  --package-type Image --code ImageUri=$IMAGE `
  --role arn:aws:iam::${ACCOUNT}:role/skillgap-lambda-role `
  --memory-size 1024 --timeout 30 --architectures x86_64 --region $REGION
```
1024 MB of memory also gives Lambda more CPU, which shortens cold starts for scikit-learn.

### 5. Create the API Gateway HTTP API
```powershell
$FN_ARN = aws lambda get-function --function-name skillgap --query Configuration.FunctionArn --output text
$API_ID = aws apigatewayv2 create-api --name skillgap-api --protocol-type HTTP `
  --target $FN_ARN --query ApiId --output text
aws lambda add-permission --function-name skillgap --statement-id apigateway-invoke `
  --action lambda:InvokeFunction --principal apigateway.amazonaws.com `
  --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT}:${API_ID}/*"
$URL = aws apigatewayv2 get-api --api-id $API_ID --query ApiEndpoint --output text
$URL
```
`--target` is "quick create": it wires every route to the Lambda and creates an auto-deploying stage.
`add-permission` allows API Gateway to call the function.

### 6. Test it
```powershell
Invoke-RestMethod "$URL/health"
Invoke-RestMethod "$URL/match" -Method Post -ContentType "application/json" `
  -Body '{"resume_text": "Python and SQL developer", "posting_text": "Need Python, SQL, AWS"}'
```
Also open `$URL/docs` in a browser. The first call after a while is slower (a "cold start"); later calls are fast.

---

## Updating after code changes (until CD is set up)
```powershell
docker build --platform linux/amd64 --provenance=false -t $IMAGE .
docker push $IMAGE
aws lambda update-function-code --function-name skillgap --image-uri $IMAGE --region $REGION
```

## Troubleshooting
- **Logs:** CloudWatch → Log groups → `/aws/lambda/skillgap`, or
  `aws logs tail /aws/lambda/skillgap --follow --region us-east-1`
- **`{"message":"Internal Server Error"}`** → check the logs; usually a Python import error.
- **Timeout on first request** → cold start; retry, or raise `--timeout`.

## Tearing it down
```powershell
aws apigatewayv2 delete-api --api-id $API_ID
aws lambda delete-function --function-name skillgap
aws ecr delete-repository --repository-name skillgap --force
aws iam detach-role-policy --role-name skillgap-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name skillgap-lambda-role
```
