@echo off
rem ============================================================
rem  Wafer-PCB-Defect-Detection  one-click launcher
rem
rem  Keep this file ASCII-only on purpose: the project lives under a
rem  path with CJK characters, and cmd's code page mangles CJK text
rem  written inside .bat files. All human-readable Chinese output and
rem  every path decision is handled by scripts\launch.py instead.
rem ============================================================
setlocal
set "HERE=%~dp0"
cd /d "%HERE%"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY="
rem 1) repo-local venv
if exist "%HERE%.venv\Scripts\python.exe" set "PY=%HERE%.venv\Scripts\python.exe"
rem 2) the interpreter this machine normally uses
if not defined PY if exist "C:\Program Files\python\python.exe" set "PY=C:\Program Files\python\python.exe"
rem 3) whatever is on PATH
if not defined PY for /f "delims=" %%I in ('where python 2^>nul') do if not defined PY set "PY=%%I"
if not defined PY for /f "delims=" %%I in ('where py 2^>nul') do if not defined PY set "PY=%%I"

if not defined PY (
  echo [ERROR] No Python found.
  echo         Install Python 3.10+ and make sure it is on PATH, then run this file again.
  echo.
  pause
  exit /b 1
)

echo Launching with: %PY%
echo.
rem Absolute script path: some shells set NoDefaultCurrentDirectoryInExePath,
rem which stops cmd from resolving anything relative to the current directory.
rem launch.py re-execs itself into a venv that has the deps, picks a free
rem port, waits for /health, then opens the browser.
"%PY%" "%HERE%scripts\launch.py" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo [ERROR] Exited with code %RC%. Read the message above.
) else (
  echo Server stopped.
)
pause
exit /b %RC%
