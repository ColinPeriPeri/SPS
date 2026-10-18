@echo off
REM ===========================================================================
REM  History store ingest.
REM
REM    run_ingest.cmd <history-file>
REM
REM  Brings the history store up to date with an export: reads the file once if
REM  it changed, then embeds every record the store has not seen. Schedule it
REM  after each export, before the robots start on tickets. Safe to re-run: an
REM  unchanged file is not read again, and an interrupted ingest resumes.
REM
REM  Exit codes:
REM    0  the store is up to date.
REM    1  Azure or the store failed part-way. Retry; it resumes.
REM    2  the file, or the configuration, needs a person to fix it.
REM ===========================================================================

setlocal

set "PROJECT_DIR=%~dp0.."

REM Accept either virtual-environment name. MIGRATION_GUIDE.md creates "venv";
REM older deployments use "env". Set SPS_PYTHON to override both.
set "PYTHON_EXE=%SPS_PYTHON%"
if "%PYTHON_EXE%"=="" if exist "%PROJECT_DIR%\venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "%PROJECT_DIR%\env\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_DIR%\env\Scripts\python.exe"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"

if "%~1"=="" (
    echo Usage: %~nx0 ^<history-file^> 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    echo Create it with: python -m venv venv 1>&2
    endlocal & exit /b 1
)

pushd "%PROJECT_DIR%"

"%PYTHON_EXE%" sps.py ingest --history "%~1"
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
