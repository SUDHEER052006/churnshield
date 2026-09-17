# Starts the ChurnShield UI on http://localhost:5173
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\frontend"

if (-not (Test-Path ".\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "UI on http://localhost:5173 (proxying /api to port 8000)" -ForegroundColor Green
npm run dev
