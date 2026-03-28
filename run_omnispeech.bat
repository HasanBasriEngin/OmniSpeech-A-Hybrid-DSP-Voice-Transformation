@echo off
setlocal

cd /d "%~dp0"

set "PY_CMD="

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 --version >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 (
        python --version >nul 2>&1
        if %errorlevel%==0 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    for %%P in (
        "%LocalAppData%\Programs\Python\Python313\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python311\python.exe"
        "%LocalAppData%\Programs\Python\Python310\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) do (
        if exist %%~P (
            set "PY_CMD=%%~P"
            goto :python_found
        )
    )
)

if not defined PY_CMD (
    echo [ERROR] Calisabilir Python bulunamadi.
    echo [INFO] Cozum: Python 3.10+ yukleyin veya PATH'e ekleyin.
    echo [INFO] Not: Windows Store python alias'i kurulu Python yoksa bu hatayi verebilir.
    pause
    exit /b 1
)

:python_found
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Sanal ortam olusturuluyor...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Sanal ortam aktive edilemedi.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt bulunamadi.
    pause
    exit /b 1
)

set "REQ_HASH_FILE=.venv\requirements.sha256"
set "NEW_HASH="
set "OLD_HASH="
set "NEED_INSTALL=1"

for /f "skip=1 tokens=1" %%H in ('certutil -hashfile requirements.txt SHA256 ^| findstr /R /V "hash CertUtil"') do (
    if not defined NEW_HASH set "NEW_HASH=%%H"
)

if exist "%REQ_HASH_FILE%" (
    set /p OLD_HASH=<"%REQ_HASH_FILE%"
)

if defined NEW_HASH if /i "%NEW_HASH%"=="%OLD_HASH%" (
    set "NEED_INSTALL=0"
)

if "%NEED_INSTALL%"=="0" (
    python -m pip check >nul 2>&1
    if errorlevel 1 (
        set "NEED_INSTALL=1"
    )
)

if "%NEED_INSTALL%"=="1" (
    echo [INFO] Eksik/guncel olmayan bagimliliklar tespit edildi. Kurulum yapiliyor...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Bagimlilik kurulumu basarisiz.
        pause
        exit /b 1
    )
    if defined NEW_HASH (
        >"%REQ_HASH_FILE%" echo %NEW_HASH%
    )
) else (
    echo [INFO] Bagimliliklar zaten hazir. Kurulum adimi atlandi.
)

echo [INFO] OmniSpeech baslatiliyor...
python main.py

if errorlevel 1 (
    echo [ERROR] Uygulama calisirken hata olustu.
    pause
    exit /b 1
)

endlocal
