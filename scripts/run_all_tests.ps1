param(
  [string]$PythonExe
)

$ErrorActionPreference = "Stop"

$python = $PythonExe
if (-not $python) {
  if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
    $python = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
  }
  elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
  }
  elseif (Test-Path ".\venv\Scripts\python.exe") {
    $python = ".\venv\Scripts\python.exe"
  }
  else {
    $python = "python"
  }
}

try {
  & $python -c "import pytest" *> $null
}
catch {
  throw "pytest is not installed in '$python'. Activate your existing venv in PowerShell (or pass -PythonExe), then install dev deps once: $python -m pip install -r requirements-dev.txt"
}

& $python -m pytest -q -m "not integration"
& $python -m pytest -q -m integration
npm test -- --run
$env:PYTHON_EXE = $python
npm run test:e2e
