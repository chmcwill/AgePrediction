param(
  [string]$StackName = "agepred-serverless",
  [string]$ConfigPath = "static/config.json"
)

$ErrorActionPreference = "Stop"

$outputs = aws cloudformation describe-stacks --stack-name $StackName --query "Stacks[0].Outputs" | ConvertFrom-Json
$predictUrl = ($outputs | Where-Object { $_.OutputKey -eq "ApiBaseUrl" }).OutputValue

if (-not $predictUrl) {
  throw "ApiBaseUrl not found in stack outputs."
}

$buildVersion = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$configContent = "{`n  `"apiBase`": `"$predictUrl`",`n  `"buildVersion`": `"$buildVersion`"`n}`n"
Set-Content -Path $ConfigPath -Value $configContent -NoNewline

Write-Host "Updated $ConfigPath with API base URL: $predictUrl"
