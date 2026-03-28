@echo off
setlocal
cd /d "%~dp0"

set "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1"
set "DOTNET_CLI_HOME=%~dp0"

dotnet run --project OmniSpeech_WPF.csproj
if errorlevel 1 (
  echo [ERROR] WPF frontend baslatilamadi.
  pause
  exit /b 1
)
