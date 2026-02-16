param(
  [string]$Region,
  [string]$BucketPrefix,
  [string]$AccountId,
  [string]$UploadBucket,
  [string]$ResultsBucket,
  [int]$PresignExpireSeconds = 600,
  [int]$ResultUrlExpireSeconds = 3600,
  [int]$BackendPort = 5000,
  [int]$StaticPort = 8000,
  [switch]$UseFlaskCli,
  [switch]$SkipUpdateConfig,
  [switch]$LocalStorage,
  [switch]$SkipCacheBust
)

$ErrorActionPreference = "Stop"

$configPath = Join-Path $PSScriptRoot "..\deploy.config.json"
if (-not $Region -or -not $BucketPrefix) {
  if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
    if (-not $Region) {
      $Region = $config.region
    }
    if (-not $BucketPrefix) {
      $BucketPrefix = $config.parameters.BucketPrefix
    }
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($LocalStorage) {
  $env:LOCAL_STORAGE = "1"
  $env:LOCAL_STORAGE_DIR = (Join-Path $repoRoot "tmp")
  New-Item -ItemType Directory -Force -Path $env:LOCAL_STORAGE_DIR | Out-Null
} else {
  if (-not $AccountId -and (-not $UploadBucket -or -not $ResultsBucket)) {
    try {
      $accountJson = aws sts get-caller-identity | ConvertFrom-Json
      $AccountId = $accountJson.Account
    } catch {
      Write-Host "Could not determine AWS account id. Pass -AccountId or -UploadBucket/-ResultsBucket."
    }
  }

  if (-not $UploadBucket -and $BucketPrefix -and $AccountId -and $Region) {
    $UploadBucket = "$BucketPrefix-uploads-$AccountId-$Region"
  }
  if (-not $ResultsBucket -and $BucketPrefix -and $AccountId -and $Region) {
    $ResultsBucket = "$BucketPrefix-results-$AccountId-$Region"
  }

  if (-not $Region -or -not $UploadBucket -or -not $ResultsBucket) {
    throw "Missing required values. Provide -Region, -UploadBucket, and -ResultsBucket (or -AccountId with a valid deploy.config.json)."
  }

  $env:S3_REGION = $Region
  $env:S3_UPLOAD_BUCKET = $UploadBucket
  $env:S3_RESULTS_BUCKET = $ResultsBucket
  $env:S3_PRESIGN_EXPIRES_SECONDS = "$PresignExpireSeconds"
  $env:S3_RESULT_URL_EXPIRES_SECONDS = "$ResultUrlExpireSeconds"
}

if (-not $SkipUpdateConfig) {
  $configFile = Join-Path $PSScriptRoot "..\static\config.json"
  $payload = @{
    apiBase      = "http://localhost:$BackendPort"
    buildVersion = "local-dev"
  } | ConvertTo-Json
  Set-Content -Path $configFile -Value $payload -NoNewline
  Write-Host "Updated $configFile with local api base."
}

if (-not $SkipCacheBust) {
  $cacheBust = Get-Date -Format "yyyyMMddHHmmss"
  $indexFile = Join-Path $PSScriptRoot "..\static\index.html"
  if (Test-Path $indexFile) {
    $content = Get-Content -Path $indexFile -Raw
    $content = $content -replace 'css/main\.css(\?v=[^"''>]*)?', "css/main.css?v=$cacheBust"
    $content = $content -replace 'js/upload\.js(\?v=[^"''>]*)?', "js/upload.js?v=$cacheBust"
    Set-Content -Path $indexFile -Value $content -NoNewline
    Write-Host "Updated cache-busting query strings in $indexFile."
  }
}

Write-Host "Starting static server on http://localhost:$StaticPort/static/index.html"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$repoRoot`"; python -m http.server $StaticPort"

Write-Host "Starting backend on http://localhost:$BackendPort"
Push-Location $repoRoot
$env:PYTHONPATH = $repoRoot
if ($UseFlaskCli) {
  $env:FLASK_APP = "age_prediction.app"
  $env:FLASK_ENV = "development"
  flask run --port $BackendPort
} else {
  python -m age_prediction.app
}
Pop-Location
