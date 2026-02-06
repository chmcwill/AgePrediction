# Serverless Deploy (CloudFormation + SAM Template)

## Prereqs
- AWS CLI configured (`aws configure`) with access to create IAM, S3, Lambda, CloudFront.
- Docker running (required to build the Lambda container image).

## Build + Deploy
1) Build and push the Lambda image (manual ImageUri path):
```bash
docker buildx build --platform linux/amd64 -t agepred-predict-age -f Dockerfile.lambda . --load --provenance=false
aws ecr create-repository --repository-name agepred-predict-age --region us-east-2
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 555813168261.dkr.ecr.us-east-2.amazonaws.com
docker tag agepred-predict-age:latest 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag>
docker push 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag>
```

2) If a previous deploy failed and stack is stuck in `ROLLBACK_COMPLETE`, delete it:
```bash
aws cloudformation delete-stack --stack-name agepred-serverless --region us-east-2
aws cloudformation wait stack-delete-complete --stack-name agepred-serverless --region us-east-2
```

If stack deletion fails with `DELETE_FAILED` (commonly due to non-empty S3 buckets), empty the buckets and retry delete:
```bash
aws s3 rm s3://agepred-uploads-555813168261-us-east-2 --recursive
aws s3 rm s3://agepred-results-555813168261-us-east-2 --recursive
aws cloudformation delete-stack --stack-name agepred-serverless --region us-east-2
aws cloudformation wait stack-delete-complete --stack-name agepred-serverless --region us-east-2
```

3) Deploy with ImageUri (short command via config):
```bash
.\scripts\deploy_stack.ps1 -ImageUri 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag>
```

Tag note:
- Local deploys: use a new tag each deploy (`v1`, `v2`, `test-20260205-1`, etc.).
- CI/CD: use the Git commit SHA as the image tag (or deploy by digest) so CloudFormation always detects a change and updates Lambda to the new image.

Note: We are not using `sam build`/`sam deploy` in this flow because SAM repeatedly failed to inject `ImageUri`
for the image-based Lambda in this project (`PredictFunction`), which caused change-set validation failures.
Building/pushing the image explicitly and deploying with CloudFormation + `ImageUri` is deterministic and reliable.
Default deploy settings live in `deploy.config.json`.

## Get Stack Outputs
```bash
aws cloudformation describe-stacks --stack-name agepred-serverless --region us-east-2 --query "Stacks[0].Outputs"
```
You need:
- `ApiBaseUrl`

## Configure the Frontend API URL
The frontend now reads `static/config.json` for the API base URL.
Run the helper script to update it after each deploy (API Gateway URL changes per stack):
```powershell
.\scripts\update_frontend.ps1 -StackName agepred-serverless
```

Note: `static/index.html` references `js/upload.js?v=1` for cache busting. When you
change frontend JS, bump the query string value if you want browsers to fetch the new file.

## Upload the Static Site
```bash
aws s3 sync static s3://<FrontendBucketName> --delete
```

Note: `FrontendBucketName`/CloudFront are added in a later template step.

## HEIC/HEIF Uploads
HEIC/HEIF files are converted to JPG in the browser via `heic2any` (CDN). The backend
does not require `libheif` in Lambda, which keeps the image build simpler.

## Retention
The SAM template uses an S3 lifecycle rule on the results bucket to expire objects after 1 day.

S3 lifecycle expiration cannot be shorter than 1 day. If you want ~1 hour retention later,
you can add a scheduled cleanup Lambda.
