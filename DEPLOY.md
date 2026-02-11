# Serverless Deploy (CloudFormation + SAM Template)

## Prereqs
- AWS CLI configured (`aws configure`) with access to create IAM, S3, Lambda, CloudFront.
- Docker running (required to build the Lambda container image).

## Quickly view UI change
```
python -m http.server 8000
http://localhost:8000/static/index.html
```

## Build + Deploy
1) Build and push the Lambda image (manual ImageUri path):
```bash
docker buildx build --platform linux/amd64 -t agepred-predict-age -f Dockerfile.lambda . --load --provenance=false
aws ecr create-repository --repository-name agepred-predict-age --region us-east-2
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 555813168261.dkr.ecr.us-east-2.amazonaws.com
docker tag agepred-predict-age:latest 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag>
docker push 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag>
```
Note: `aws ecr create-repository` is a one-time setup; skip it after the repo exists.

2) All-in-one reset + deploy (run after step 1):
```powershell
.\scripts\redeploy_all.ps1 -ImageUri 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:<tag> #base
.\scripts\redeploy_all.ps1 -ImageUri 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:v3 -SkipDelete -OpenFrontend #if updated backend
.\scripts\redeploy_all.ps1 -ImageUri 555813168261.dkr.ecr.us-east-2.amazonaws.com/agepred-predict-age:v3 -SkipDelete -SkipDeployStack -OpenFrontend #if updated frontend
```
Notes:
- This deletes the stack, empties the buckets, redeploys, updates `static/config.json`, syncs `static/`, and invalidates CloudFront.
- Add `-Force` to skip the destructive delete confirmation prompt.
- Add `-SkipDelete` for most updates. You generally only need a full delete if the stack is stuck in `ROLLBACK_COMPLETE`/`DELETE_FAILED` or you changed immutable properties (like bucket names) that require replacement.
- Add `-SkipFrontendUpdate` when only backend code changed.
- Add `-SkipFrontendSync` when only backend code changed.
- Add `-SkipInvalidateCloudFront` if you want to avoid cache invalidations.
- Add `-OpenFrontend` to open the CloudFront URL after deploy.
- Add `-SkipDeployStack` if you only want to update/sync the frontend without touching CloudFormation.

If you want to run components instead of the all-in-one script, use the sections below.

3) If a previous deploy failed and stack is stuck in `ROLLBACK_COMPLETE`, delete it (use only when the stack is broken):
```bash
aws cloudformation delete-stack --stack-name agepred-serverless --region us-east-2
aws cloudformation wait stack-delete-complete --stack-name agepred-serverless --region us-east-2
```

If stack deletion fails with `DELETE_FAILED` (commonly due to non-empty S3 buckets), empty the buckets and retry delete:
```bash
aws s3 rm s3://agepred-uploads-555813168261-us-east-2 --recursive
aws s3 rm s3://agepred-results-555813168261-us-east-2 --recursive
aws s3 rm s3://agepred-frontend-555813168261-us-east-2 --recursive
aws cloudformation delete-stack --stack-name agepred-serverless --region us-east-2
aws cloudformation wait stack-delete-complete --stack-name agepred-serverless --region us-east-2
```

4) Deploy with ImageUri (short command via config) (use when backend code or template changed):
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
- `FrontendBucketName`
- `CloudFrontUrl`
- `CloudFrontDistributionId` (for invalidations)

## Configure the Frontend API URL
The frontend now reads `static/config.json` for the API base URL.
Run the helper script to update it after each deploy (API Gateway URL changes per stack):
```powershell
.\scripts\update_frontend.ps1 -StackName agepred-serverless
```
Use this when you only changed frontend config or API base.

Note: `static/index.html` references `js/upload.js?v=1` for cache busting. When you
change frontend JS, bump the query string value if you want browsers to fetch the new file.

## Upload the Static Site
```bash
aws s3 sync static s3://<FrontendBucketName> --delete
e.g. aws s3 sync static s3://agepred-frontend-555813168261-us-east-2 --delete
```
Use this when you changed HTML/CSS/JS or `static/config.json`.

You can print (or open) the CloudFront URL:
```powershell
.\scripts\open_frontend.ps1 -StackName agepred-serverless
.\scripts\open_frontend.ps1 -StackName agepred-serverless -Open
```

If you want new frontend changes to appear immediately, invalidate CloudFront:
```bash
aws cloudfront create-invalidation --distribution-id <CloudFrontDistributionId> --paths "/*"
```
Use this when you need the CDN to serve updated files right away.

## HEIC/HEIF Uploads
HEIC/HEIF files are converted to JPG in the browser via `heic2any` (CDN). The backend does not require `libheif` in Lambda, which keeps the image build simpler.

## Quick Cold Start Test
To force a cold start without rebuilding/pushing an image, bump the Lambda timeout (or any config value).
This forces Lambda to spin up new containers on the next request.

```bash
aws lambda update-function-configuration --function-name predict_age --region us-east-2 --timeout 61
```
Make one request (page load + submit) to observe the cold-start warmup message.
Optionally revert:
```bash
aws lambda update-function-configuration --function-name predict_age --region us-east-2 --timeout 60
```
## Retention
The template uses S3 lifecycle rules on uploads and results buckets to expire objects after 1 day.

S3 lifecycle expiration cannot be shorter than 1 day. If you want ~1 hour retention later, you can add a scheduled cleanup Lambda.
