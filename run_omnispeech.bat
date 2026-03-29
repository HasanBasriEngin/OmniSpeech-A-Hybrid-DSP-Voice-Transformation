@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not defined OMNISPEECH_AUTO_INSTALL set "OMNISPEECH_AUTO_INSTALL=1"
if not defined OMNISPEECH_AUTO_INSTALL_MSVC set "OMNISPEECH_AUTO_INSTALL_MSVC=1"
call :refresh_path

:detect_python
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
    if not defined PY_CMD if exist %%~P (
      set "PY_CMD=%%~P"
    )
  )
)

if not defined PY_CMD (
  if /i "!OMNISPEECH_AUTO_INSTALL:~0,1!"=="1" (
    if not defined _PY_AUTO_INSTALL_TRIED (
      set "_PY_AUTO_INSTALL_TRIED=1"
      call :auto_install_python
      call :refresh_path
      goto :detect_python
    )
  )

  echo [ERROR] Working Python 3.10+ not found.
  echo [INFO] Install Python and ensure PATH is configured.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating Python virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create .venv.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Could not activate .venv.
  pause
  exit /b 1
)

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found.
  pause
  exit /b 1
)

set "NEED_PY_INSTALL=0"
if not exist ".venv\.deps_ready" set "NEED_PY_INSTALL=1"
if /i "!OMNISPEECH_FORCE_INSTALL:~0,1!"=="1" set "NEED_PY_INSTALL=1"

if "!NEED_PY_INSTALL!"=="0" (
  python -m pip check >nul 2>&1
  if errorlevel 1 set "NEED_PY_INSTALL=1"
)

if "!NEED_PY_INSTALL!"=="1" (
  echo [INFO] Installing Python dependencies...
  python -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Python dependency installation failed.
    pause
    exit /b 1
  )
  >".venv\.deps_ready" echo ok
) else (
  echo [INFO] Python dependencies are up to date.
)

:detect_npm
where npm >nul 2>&1
if errorlevel 1 (
  if /i "!OMNISPEECH_AUTO_INSTALL:~0,1!"=="1" (
    if not defined _NODE_AUTO_INSTALL_TRIED (
      set "_NODE_AUTO_INSTALL_TRIED=1"
      call :auto_install_node
      call :refresh_path
      goto :detect_npm
    )
  )

  echo [ERROR] npm not found. Install Node.js 20+ first.
  pause
  exit /b 1
)

if not exist "package.json" (
  echo [ERROR] package.json not found.
  pause
  exit /b 1
)

set "NEED_NPM_INSTALL=0"
if not exist "node_modules" set "NEED_NPM_INSTALL=1"
if /i "!OMNISPEECH_FORCE_INSTALL:~0,1!"=="1" set "NEED_NPM_INSTALL=1"

if "!NEED_NPM_INSTALL!"=="1" (
  echo [INFO] Installing npm dependencies...
  cmd /c npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
) else (
  echo [INFO] npm dependencies are up to date.
)

set "ESBUILD_EXE=node_modules\@esbuild\win32-x64\esbuild.exe"
if exist "!ESBUILD_EXE!" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Unblock-File -Path 'node_modules\\@esbuild\\win32-x64\\esbuild.exe' -ErrorAction Stop } catch { }" >nul 2>&1
  cmd /c "!ESBUILD_EXE! --version" >nul 2>&1
  if errorlevel 1 (
    echo [INFO] Repairing esbuild runtime...
    cmd /c npm rebuild esbuild
    if errorlevel 1 (
      echo [ERROR] esbuild rebuild failed.
      pause
      exit /b 1
    )
  )
)

if /i "!OMNISPEECH_SETUP_ONLY:~0,1!"=="1" (
  echo [INFO] Setup completed. Skipping launch because OMNISPEECH_SETUP_ONLY=1.
  endlocal
  exit /b 0
)

:detect_cargo
where cargo >nul 2>&1
if errorlevel 1 (
  if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;!PATH!"
  )
)

where cargo >nul 2>&1
if errorlevel 1 (
  if /i "!OMNISPEECH_AUTO_INSTALL:~0,1!"=="1" (
    if not defined _RUST_AUTO_INSTALL_TRIED (
      set "_RUST_AUTO_INSTALL_TRIED=1"
      call :auto_install_rust
      call :refresh_path
      goto :detect_cargo
    )
  )

  echo [ERROR] cargo not found. Desktop mode requires Tauri + Rust toolchain.
  echo [INFO] Try: winget install -e --id Rustlang.Rustup
  echo [INFO] Install Rust from: https://rustup.rs/
  pause
  exit /b 1
)

echo [INFO] Verifying Rust toolchain...
cargo --version >nul 2>&1
if errorlevel 1 (
  where rustup >nul 2>&1
  if not errorlevel 1 (
    rustup default stable
  )
)

cargo --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] cargo exists but Rust toolchain is not configured correctly.
  echo [INFO] Run: rustup default stable
  pause
  exit /b 1
)

call :load_vsdevcmd

where link >nul 2>&1
if errorlevel 1 (
  if /i "!OMNISPEECH_AUTO_INSTALL_MSVC:~0,1!"=="1" (
    if not defined _MSVC_AUTO_INSTALL_TRIED (
      set "_MSVC_AUTO_INSTALL_TRIED=1"
      call :auto_install_msvc
      call :load_vsdevcmd
    )
  )
)

where link >nul 2>&1
if errorlevel 1 (
  echo [ERROR] MSVC linker link.exe not found.
  echo [INFO] Install Visual Studio Build Tools with C++ workload.
  echo [INFO] Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
  echo [INFO] Required components:
  echo [INFO]   - Desktop development with C++
  echo [INFO]   - MSVC v143 x64/x86 build tools
  echo [INFO]   - Windows 10/11 SDK
  pause
  exit /b 1
)

call :free_port 1420
if errorlevel 1 (
  pause
  exit /b 1
)

echo [INFO] Launching OmniSpeech - Tauri dev...
cmd /c npm run tauri dev

if errorlevel 1 (
  echo [ERROR] Launch failed.
  pause
  exit /b 1
)

endlocal
exit /b 0

:refresh_path
set "PATH=%USERPROFILE%\.cargo\bin;%ProgramFiles%\nodejs;%LocalAppData%\Microsoft\WindowsApps;%PATH%"
for %%P in (
  "%LocalAppData%\Programs\Python\Python313"
  "%LocalAppData%\Programs\Python\Python312"
  "%LocalAppData%\Programs\Python\Python311"
  "%LocalAppData%\Programs\Python\Python310"
) do (
  if exist "%%~P\python.exe" set "PATH=%%~P;%%~P\Scripts;!PATH!"
)
exit /b 0

:ensure_winget
where winget >nul 2>&1
if errorlevel 1 (
  echo [WARN] winget not found. Automatic installation is limited.
  exit /b 1
)
exit /b 0

:auto_install_python
echo [INFO] Python 3.10+ not found. Attempting automatic install...
call :ensure_winget
if errorlevel 1 exit /b 1

winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo [WARN] Python 3.12 install failed, trying Python 3.11...
  winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
)

if errorlevel 1 (
  echo [WARN] Automatic Python install failed.
  exit /b 1
)

echo [INFO] Python installation completed.
exit /b 0

:auto_install_node
echo [INFO] Node.js (npm) not found. Attempting automatic install...
call :ensure_winget
if errorlevel 1 exit /b 1

winget install -e --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo [WARN] Automatic Node.js install failed.
  exit /b 1
)

echo [INFO] Node.js installation completed.
exit /b 0

:auto_install_rust
echo [INFO] Rust toolchain not found. Attempting automatic install...
call :ensure_winget
if not errorlevel 1 (
  winget install -e --id Rustlang.Rustup --silent --accept-package-agreements --accept-source-agreements
)

if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
  set "PATH=%USERPROFILE%\.cargo\bin;!PATH!"
)

where cargo >nul 2>&1
if errorlevel 1 (
  echo [INFO] winget path failed, trying rustup bootstrap installer...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri https://win.rustup.rs -OutFile \"$env:TEMP\\rustup-init.exe\" -UseBasicParsing } catch { exit 1 }"
  if exist "%TEMP%\rustup-init.exe" (
    "%TEMP%\rustup-init.exe" -y --default-toolchain stable
  )
)

if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
  set "PATH=%USERPROFILE%\.cargo\bin;!PATH!"
)

where rustup >nul 2>&1
if not errorlevel 1 rustup default stable >nul 2>&1

where cargo >nul 2>&1
if errorlevel 1 (
  echo [WARN] Automatic Rust installation failed.
  exit /b 1
)

echo [INFO] Rust installation completed.
exit /b 0

:auto_install_msvc
echo [INFO] MSVC linker not found. Attempting automatic Visual Studio Build Tools install...
call :ensure_winget
if not errorlevel 1 (
  winget install -e --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--passive --wait --norestart --nocache --includeRecommended --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041"
)

where link >nul 2>&1
if not errorlevel 1 (
  echo [INFO] MSVC tools are available.
  exit /b 0
)

if not exist "%TEMP%\vs_BuildTools.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vs_BuildTools.exe -OutFile \"$env:TEMP\\vs_BuildTools.exe\" -UseBasicParsing } catch { exit 1 }"
)

if exist "%TEMP%\vs_BuildTools.exe" (
  echo [INFO] Running Build Tools installer directly...
  "%TEMP%\vs_BuildTools.exe" --passive --wait --norestart --nocache --includeRecommended --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041
)

exit /b 0

:load_vsdevcmd
where link >nul 2>&1
if not errorlevel 1 exit /b 0

set "VSDEV_CMD="
set "VSWHERE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "!VSWHERE_EXE!" (
  for /f "usebackq delims=" %%I in (`"!VSWHERE_EXE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if not defined VSDEV_CMD if exist "%%~I\Common7\Tools\VsDevCmd.bat" set "VSDEV_CMD=%%~I\Common7\Tools\VsDevCmd.bat"
  )
)

for %%F in (
  "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
) do (
  if not defined VSDEV_CMD if exist %%~F set "VSDEV_CMD=%%~F"
)

if defined VSDEV_CMD (
  echo [INFO] Loading Visual Studio C++ build environment...
  call "!VSDEV_CMD!" -arch=x64 -host_arch=x64 >nul
)

exit /b 0

:free_port
set "TARGET_PORT=%~1"
set "PORT_BUSY="

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%TARGET_PORT% .*LISTENING"') do (
  if not "%%P"=="0" (
    set "PORT_BUSY=1"
    echo [WARN] Port %TARGET_PORT% is in use by PID %%P. Stopping stale process...
    taskkill /PID %%P /F >nul 2>&1
  )
)

if defined PORT_BUSY (
  timeout /t 1 >nul
  netstat -ano | findstr /R /C:":%TARGET_PORT% .*LISTENING" >nul 2>&1
  if not errorlevel 1 (
    echo [ERROR] Port %TARGET_PORT% is still in use.
    echo [INFO] Run: netstat -ano ^| findstr :%TARGET_PORT%
    echo [INFO] Then: taskkill /PID ^<PID^> /F
    exit /b 1
  )
)

exit /b 0
