# Starts the ChurnShield API. Builds artifacts first if they are missing.
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\backend"

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No virtualenv found. Creating one..." -ForegroundColor Yellow
    py -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

if (-not (Test-Path ".\artifacts\summary.json")) {
    Write-Host "Artifacts missing - running the pipeline (this takes a few minutes)" -ForegroundColor Cyan
    & $py -m src.build_artifacts
}

Write-Host "API on http://127.0.0.1:8000  (docs at /docs)" -ForegroundColor Green
& $py -m uvicorn app.main:app --reload --port 8000
