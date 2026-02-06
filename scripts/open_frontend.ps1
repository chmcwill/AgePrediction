param(
  [string]$StackName = "agepred-serverless",
  [switch]$Open
)

$ErrorActionPreference = "Stop"

$outputs = aws cloudformation describe-stacks --stack-name $StackName --query "Stacks[0].Outputs" | ConvertFrom-Json
$frontendUrl = ($outputs | Where-Object { $_.OutputKey -eq "CloudFrontUrl" }).OutputValue

if (-not $frontendUrl) {
  throw "CloudFrontUrl not found in stack outputs."
}

Write-Host "Frontend URL: $frontendUrl"

if ($Open) {
  Start-Process $frontendUrl
}
