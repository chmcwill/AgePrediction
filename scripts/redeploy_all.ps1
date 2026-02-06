param(
  [Parameter(Mandatory = $true)]
  [string]$ImageUri,
  [switch]$SkipDelete,
  [switch]$SkipFrontendUpdate,
  [switch]$Force,
  [switch]$OpenFrontend
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
  Write-Host "Deleting stack '$stackName' in $region..."
  aws cloudformation delete-stack --stack-name $stackName --region $region | Out-Null
  try {
    aws cloudformation wait stack-delete-complete --stack-name $stackName --region $region | Out-Null
  } catch {
    Write-Host "Stack deletion failed or is blocked. Attempting to empty buckets..."
    .\scripts\empty_buckets.ps1 -StackName $stackName -Region $region -BucketPrefix $bucketPrefix -Force:$Force
    Write-Host "Retrying stack delete..."
    aws cloudformation delete-stack --stack-name $stackName --region $region | Out-Null
    aws cloudformation wait stack-delete-complete --stack-name $stackName --region $region | Out-Null
  }
}

Write-Host "Deploying stack with ImageUri: $ImageUri"
.\scripts\deploy_stack.ps1 -ImageUri $ImageUri

if (-not $SkipFrontendUpdate) {
  Write-Host "Updating frontend config.json with API base URL..."
  .\scripts\update_frontend.ps1 -StackName $stackName
}

if ($OpenFrontend) {
  .\scripts\open_frontend.ps1 -StackName $stackName -Open
}

Write-Host "Redeploy complete."
