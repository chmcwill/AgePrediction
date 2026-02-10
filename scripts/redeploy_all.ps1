param(
  [Parameter(Mandatory = $true)]
  [string]$ImageUri,
  [switch]$SkipDelete,
  [switch]$SkipFrontendUpdate,
  [switch]$Force,
  [switch]$OpenFrontend,
  [switch]$SkipFrontendSync,
  [switch]$SkipInvalidateCloudFront,
  [switch]$SkipDeployStack
)

$ErrorActionPreference = "Stop"

$configPath = "deploy.config.json"
if (-not (Test-Path -Path $configPath)) {
  throw "deploy.config.json not found."
}

$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
$stackName = $config.stack_name
$region = $config.region
$bucketPrefix = $config.parameters.BucketPrefix

if (-not $SkipDelete) {
  Write-Host "Emptying buckets before delete..."
  .\scripts\empty_buckets.ps1 -StackName $stackName -Region $region -BucketPrefix $bucketPrefix -Force:$Force
  Write-Host "Deleting stack '$stackName' in $region..."
  aws cloudformation delete-stack --stack-name $stackName --region $region | Out-Null
  aws cloudformation wait stack-delete-complete --stack-name $stackName --region $region | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "Stack deletion failed. Resolve DELETE_FAILED before redeploying."
  }
}

if (-not $SkipDeployStack) {
  Write-Host "Deploying stack with ImageUri: $ImageUri"
  .\scripts\deploy_stack.ps1 -ImageUri $ImageUri
}

if (-not $SkipFrontendUpdate) {
  Write-Host "Updating frontend config.json with API base URL..."
  .\scripts\update_frontend.ps1 -StackName $stackName
}

if (-not $SkipFrontendSync) {
  $outputs = aws cloudformation describe-stacks --stack-name $stackName --query "Stacks[0].Outputs" | ConvertFrom-Json
  $frontendBucket = ($outputs | Where-Object { $_.OutputKey -eq "FrontendBucketName" }).OutputValue
  if (-not $frontendBucket) {
    throw "FrontendBucketName not found in stack outputs."
  }
  Write-Host "Syncing static site to s3://$frontendBucket ..."
  aws s3 sync static "s3://$frontendBucket" --delete | Out-Null
}

if (-not $SkipInvalidateCloudFront) {
  $outputs = aws cloudformation describe-stacks --stack-name $stackName --query "Stacks[0].Outputs" | ConvertFrom-Json
  $distId = ($outputs | Where-Object { $_.OutputKey -eq "CloudFrontDistributionId" }).OutputValue
  if (-not $distId) {
    throw "CloudFrontDistributionId not found in stack outputs."
  }
  Write-Host "Creating CloudFront invalidation for distribution $distId..."
  aws cloudfront create-invalidation --distribution-id $distId --paths "/*" | Out-Null
}

if ($OpenFrontend) {
  .\scripts\open_frontend.ps1 -StackName $stackName -Open
}

Write-Host "Redeploy complete."
