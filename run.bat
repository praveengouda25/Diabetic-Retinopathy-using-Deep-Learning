@echo off
REM ============================================
REM Diabetic Retinopathy Detection - Run Script
REM ============================================
REM This script activates the virtual environment and runs the Gradio web app.
REM The app will be accessible at http://127.0.0.1:7860

echo.
echo ============================================
echo  Starting Diabetic Retinopathy Detection App
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the Gradio app
echo Starting the web app...
echo.
echo The app will open at: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py

pause

