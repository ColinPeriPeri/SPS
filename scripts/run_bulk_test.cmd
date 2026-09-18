@echo off
REM ===========================================================================
REM  Bulk test runner.
REM
REM    run_bulk_test.cmd <tickets-file> <history-file> [output-file]
REM
REM  Resolves every row of <tickets-file> through the same pipeline the UiPath
REM  Performer calls, and writes a COPY carrying every original column plus the
REM  result columns. The input workbook is never modified.
REM
REM  Default output is <tickets>_results.xlsx beside the input.
REM
REM  Prefer a .csv history. resolve() re-scans it once per ticket, so a large
REM  .xlsx history dominates the run: the same 300k rows take ~1.5 s as CSV and
REM  ~40 s as .xlsx, per ticket.
REM
REM  Exit codes:
REM    0  the sheet ran. Individual rows may have legitimately failed.
REM    1  at least one row hit an infrastructure error.
REM    2  the inputs could not be read.
REM ===========================================================================

setlocal

set "PROJECT_DIR=%~dp0.."

REM Accept either virtual-environment name. MIGRATION_GUIDE.md creates "venv";
REM older deployments use "env". Set SPS_PYTHON to override both.
set "PYTHON_EXE=%SPS_PYTHON%"
if "%PYTHON_EXE%"=="" if exist "%PROJECT_DIR%\venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "%PROJECT_DIR%\env\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_DIR%\env\Scripts\python.exe"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"

if "%~2"=="" (
    echo Usage: %~nx0 ^<tickets-file^> ^<history-file^> [output-file] 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    echo Create it with: python -m venv venv 1>&2
    endlocal & exit /b 1
)

pushd "%PROJECT_DIR%"

if "%~3"=="" (
    "%PYTHON_EXE%" -m scripts.run_bulk_test --tickets "%~1" --history "%~2"
) else (
    "%PYTHON_EXE%" -m scripts.run_bulk_test --tickets "%~1" --history "%~2" --output "%~3"
)
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
