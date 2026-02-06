param(
  [string]$StackName = "",
  [string]$Region = "",
  [string]$BucketPrefix = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$configPath = "deploy.config.json"
if ((-not $StackName) -or (-not $Region) -or (-not $BucketPrefix)) {
  if (Test-Path -Path $configPath) {
    $config = Get-Content -Raw -Path $configPath | ConvertFrom-Json
    if (-not $StackName) { $StackName = $config.stack_name }
    if (-not $Region) { $Region = $config.region }
    if (-not $BucketPrefix) { $BucketPrefix = $config.parameters.BucketPrefix }
  }
}

if (-not $StackName -or -not $Region -or -not $BucketPrefix) {
  throw "StackName, Region, and BucketPrefix are required (pass params or ensure deploy.config.json exists)."
}

$accountId = (aws sts get-caller-identity --query Account --output text).Trim()
if (-not $accountId) {
  throw "Unable to resolve AWS account id."
}

$bucketNames = @(
  "$BucketPrefix-uploads-$accountId-$Region",
  "$BucketPrefix-results-$accountId-$Region",
  "$BucketPrefix-frontend-$accountId-$Region"
)

if (-not $Force) {
  Write-Host "This will delete ALL objects in the following buckets:"
  $bucketNames | ForEach-Object { Write-Host "  - $_" }
  $confirm = Read-Host "Type 'DELETE' to continue"
  if ($confirm -ne "DELETE") {
    throw "Aborted by user."
  }
}

foreach ($bucket in $bucketNames) {
  Write-Host "Emptying s3://$bucket ..."
  aws s3 rm "s3://$bucket" --recursive | Out-Null
}

Write-Host "Bucket cleanup complete."
