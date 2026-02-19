param(
  [string]$StackName = "agepred-serverless",
  [string]$PreferredUrl = "https://facepredictionservice.com",
  [switch]$Open
)

$ErrorActionPreference = "Stop"

$frontendUrl = $PreferredUrl
if (-not $frontendUrl) {
  $outputs = aws cloudformation describe-stacks --stack-name $StackName --query "Stacks[0].Outputs" | ConvertFrom-Json
  $frontendUrl = ($outputs | Where-Object { $_.OutputKey -eq "CloudFrontUrl" }).OutputValue
}

if (-not $frontendUrl) {
  throw "Frontend URL not found."
}

Write-Host "Frontend URL: $frontendUrl"

if ($Open) {
  Start-Process $frontendUrl
}
