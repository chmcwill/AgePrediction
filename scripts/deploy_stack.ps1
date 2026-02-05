param(
  [Parameter(Mandatory = $true)]
  [string]$ImageUri,
  [string]$ConfigPath = "deploy.config.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $ConfigPath)) {
  throw "Config not found: $ConfigPath"
}

$cfg = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json

$parameterOverrides = @("ImageUri=$ImageUri")
foreach ($prop in $cfg.parameters.PSObject.Properties) {
  $parameterOverrides += "$($prop.Name)=$($prop.Value)"
}

$args = @(
  "cloudformation", "deploy",
  "--template-file", $cfg.template_file,
  "--stack-name", $cfg.stack_name,
  "--capabilities", $cfg.capabilities,
  "--region", $cfg.region,
  "--parameter-overrides"
) + $parameterOverrides

Write-Host "Deploying stack '$($cfg.stack_name)' to region '$($cfg.region)'..."
aws @args
