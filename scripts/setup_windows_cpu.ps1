param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$PipCache = Join-Path $ProjectRoot ".pip-cache"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path -LiteralPath $VirtualEnvironment)) {
    & $Python -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment (exit $LASTEXITCODE)." }
}

$VenvPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$env:PIP_CACHE_DIR = $PipCache
$env:DGLBACKEND = "pytorch"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling (exit $LASTEXITCODE)." }
& $VenvPython -m pip install --requirement $Requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU requirements (exit $LASTEXITCODE)." }
& $VenvPython (Join-Path $PSScriptRoot "check_environment.py") --allow-cpu
if ($LASTEXITCODE -ne 0) { throw "CPU environment check failed (exit $LASTEXITCODE)." }
