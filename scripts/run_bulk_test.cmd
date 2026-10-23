@echo off
REM ===========================================================================
REM  Bulk test runner.
REM
REM    run_bulk_test.cmd <tickets-file> <history-file> [output-file] [commodity-groups-file]
REM
REM  Resolves every row of <tickets-file> through the same pipeline the UiPath
REM  Performer calls, and writes a COPY carrying every original column plus the
REM  result columns. The input workbook is never modified.
REM
REM  Default output is <tickets>_results.xlsx beside the input. The fourth
REM  argument, optional, is the commodity groups .csv/.xlsx (else
REM  SPS_COMMODITY_GROUPS); give the output file too when you pass it. From
REM  cmd.exe, "" keeps the default output, but Windows PowerShell drops a ""
REM  before this script sees it, making the groups file the output -- which
REM  is refused before any ticket runs: only an earlier results sheet is
REM  ever replaced.
REM
REM  The history store is synced once, by the first row if the export changed;
REM  later rows read only their family from it. Run run_ingest.cmd first to
REM  keep the embedding out of the run.
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
    echo Usage: %~nx0 ^<tickets-file^> ^<history-file^> [output-file] [commodity-groups-file] 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    echo Create it with: python -m venv venv 1>&2
    endlocal & exit /b 1
)

pushd "%PROJECT_DIR%"

set "OUTPUT="
if not "%~3"=="" set OUTPUT=--output "%~3"
set "GROUPS="
if not "%~4"=="" set GROUPS=--commodity-file "%~4"
"%PYTHON_EXE%" sps.py bulk --tickets "%~1" --history "%~2" %OUTPUT% %GROUPS%
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
