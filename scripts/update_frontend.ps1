param(
  [string]$StackName = "agepred-serverless",
  [string]$IndexPath = "static/index.html"
)

$ErrorActionPreference = "Stop"

$outputs = aws cloudformation describe-stacks --stack-name $StackName --query "Stacks[0].Outputs" | ConvertFrom-Json
$predictUrl = ($outputs | Where-Object { $_.OutputKey -eq "ApiBaseUrl" }).OutputValue

if (-not $predictUrl) {
  throw "ApiBaseUrl not found in stack outputs."
}

$indexContent = Get-Content -Raw -Path $IndexPath
$updated = $indexContent -replace "__API_BASE_URL__", $predictUrl
Set-Content -Path $IndexPath -Value $updated -NoNewline

Write-Host "Updated $IndexPath with API base URL: $predictUrl"
