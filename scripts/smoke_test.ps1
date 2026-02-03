param(
  [string]$StackName = "agepred-serverless",
  [string]$ImagePath,
  [string]$ApiBase = ""
)

$ErrorActionPreference = "Stop"

if (-not $ImagePath) {
  throw "ImagePath is required. Example: .\scripts\smoke_test.ps1 -ImagePath .\static\images\sample.jpg"
}

if (-not (Test-Path -Path $ImagePath)) {
  throw "Image not found: $ImagePath"
}

if (-not $ApiBase) {
  $outputs = aws cloudformation describe-stacks --stack-name $StackName --query "Stacks[0].Outputs" | ConvertFrom-Json
  $ApiBase = ($outputs | Where-Object { $_.OutputKey -eq "PredictFunctionUrl" }).OutputValue
}

if (-not $ApiBase) {
  throw "API base URL not found. Pass -ApiBase or ensure stack outputs include PredictFunctionUrl."
}

$ext = [System.IO.Path]::GetExtension($ImagePath).ToLowerInvariant()
$contentType = switch ($ext) {
  ".jpg" { "image/jpeg" }
  ".jpeg" { "image/jpeg" }
  ".png" { "image/png" }
  ".webp" { "image/webp" }
  ".heic" { "image/heic" }
  ".heif" { "image/heif" }
  Default { "application/octet-stream" }
}

Write-Host "Using API base: $ApiBase"

$filename = [System.IO.Path]::GetFileName($ImagePath)
$presignBody = @{
  filename = $filename
  content_type = $contentType
} | ConvertTo-Json

$presign = Invoke-RestMethod -Method Post -Uri "$ApiBase/api/presign" -ContentType "application/json" -Body $presignBody
if (-not $presign.url) {
  throw "Presign failed: $($presign | ConvertTo-Json -Depth 5)"
}

Write-Host "Uploading image..."
Invoke-WebRequest -Method Put -Uri $presign.url -InFile $ImagePath -ContentType $contentType | Out-Null

Write-Host "Requesting prediction..."
$predictBody = @{
  key = $presign.key
} | ConvertTo-Json

$predict = Invoke-RestMethod -Method Post -Uri "$ApiBase/api/predict" -ContentType "application/json" -Body $predictBody

Write-Host "Prediction complete."
Write-Host "Big figure URL: $($predict.big_fig_url)"
Write-Host "Face figure URLs:"
$predict.fig_urls | ForEach-Object { Write-Host "  $_" }
