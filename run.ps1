# Diabetic Retinopathy Detection - PowerShell Run Script
# ======================================================
# This script activates the virtual environment and runs the Gradio web app.

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Diabetic Retinopathy Detection App" -ForegroundColor Cyan  
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

# Run the app
Write-Host ""
Write-Host "Starting the web app..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will open at: http://127.0.0.1:7860" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
python app.py

