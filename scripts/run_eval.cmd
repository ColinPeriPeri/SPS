@echo off
REM ===========================================================================
REM  Batch evaluation runner.
REM
REM    run_eval.cmd <test-dir> [output-dir] [shared-history-file]
REM
REM  Runs every ticket in <test-dir> and writes eval_results.xlsx, one row per
REM  case, carrying the raw similarity score whether or not the case cleared the
REM  gate -- which is the point: calibrating a threshold needs the scores of the
REM  cases it rejected, not just the ones it passed.
REM
REM  Test files are named <id>_ticket.csv|xlsx, optionally paired with
REM  <id>_history.csv|xlsx. Pass a third argument to use one shared history for
REM  every case, which is the usual shape.
REM
REM  The whole batch runs in one process, so the local model is loaded once
REM  rather than once per case.
REM
REM  Exit codes:
REM    0  the batch ran. Individual cases may have legitimately failed.
REM    1  at least one case hit an infrastructure error.
REM    2  the case list could not be built.
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
    echo Usage: %~nx0 ^<test-dir^> [output-dir] [shared-history-file] 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    echo Create it with: python -m venv venv 1>&2
    endlocal & exit /b 1
)

set "OUT=%~2"
if "%OUT%"=="" set "OUT=."

pushd "%PROJECT_DIR%"

if "%~3"=="" (
    "%PYTHON_EXE%" -m scripts.run_eval_batch --test-dir "%~1" --output-dir "%OUT%"
) else (
    "%PYTHON_EXE%" -m scripts.run_eval_batch --test-dir "%~1" --output-dir "%OUT%" --history-file "%~3"
)
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
