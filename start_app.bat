@echo off
echo ========================================
echo    STT Arena - Starting Application
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo [2/3] Checking dependencies...
pip list | findstr "streamlit" >nul
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

REM Check if .env exists
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    echo Please copy .env.example to .env and add your API keys.
    echo.
    echo Do you want to continue anyway? (Y/N)
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
)

REM Start Streamlit app
echo [3/3] Starting STT Arena...
echo.
echo ========================================
echo  App will open in your browser at:
echo  http://localhost:8501
echo ========================================
echo.
streamlit run app.py

pause
