# Serverless Deploy (AWS SAM)

## Prereqs
- AWS CLI configured (`aws configure`) with access to create IAM, S3, Lambda, CloudFront.
- AWS SAM CLI installed.
- Docker running (required to build the Lambda container image).

## Build + Deploy
1) Build the image-based Lambda:
```bash
sam build
```

Note: Lambda uses `requirements-lambda.txt` (API-only deps). The image does not include
templates/static assets; those are served from S3/CloudFront.

2) Deploy (first time or updates):
```bash
sam deploy
```

Defaults live in `samconfig.toml` (edit there if you want to change values).
If this is your first deploy, run:
`sam deploy --guided --resolve-image-repos`

## Get Stack Outputs
```bash
aws cloudformation describe-stacks --stack-name agepred-serverless --query "Stacks[0].Outputs"
```
You need:
- `PredictFunctionUrl`
- `FrontendBucketName`
- `FrontendDistributionDomainName` (if CloudFront enabled)

## Configure the Frontend API URL
Run the helper script to inject the Function URL into `static/index.html`:
```powershell
.\scripts\update_frontend.ps1 -StackName agepred-serverless
```

## Upload the Static Site
```bash
aws s3 sync static s3://<FrontendBucketName> --delete
```

If CloudFront is enabled, use:
```
https://<FrontendDistributionDomainName>
```

## HEIC/HEIF Uploads
HEIC/HEIF files are converted to JPG in the browser via `heic2any` (CDN). The backend
does not require `libheif` in Lambda, which keeps the image build simpler.

## Retention
The SAM template uses an S3 lifecycle rule on the results bucket to expire objects after 1 day.

S3 lifecycle expiration cannot be shorter than 1 day. If you want ~1 hour retention later,
you can add a scheduled cleanup Lambda.
