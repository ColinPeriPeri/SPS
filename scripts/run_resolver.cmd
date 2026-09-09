@echo off
REM ===========================================================================
REM  SPS resolver wrapper for the UiPath Performer.
REM
REM    run_resolver.cmd <ticket-file> <history-file> <output-dir>
REM
REM  Exit codes (propagated verbatim, so the state machine can branch without
REM  opening a workbook):
REM    0  the run completed. PASS, or a legitimate FAIL such as a gated ticket
REM       or an unknown part -- log a business exception, do not retry.
REM    1  infrastructure fault (Azure unreachable, unhandled error). Retry.
REM    2  the ticket or history file could not be read. Alert a human.
REM
REM  Writes into <output-dir>:
REM    status.xlsx  ALWAYS, including an early abort or an unhandled exception.
REM                 Execution_Timestamp, Status, Status_Code, Reason.
REM    output.xlsx  only when Status is PASS.
REM
REM  Both are cleared before work starts, so if this process is killed outright
REM  the caller finds neither: "status.xlsx missing" is unambiguous.
REM
REM  Prefer a .csv history. The same 300k rows take ~1.5 s as CSV and ~40 s as
REM  .xlsx, because openpyxl parses XML per row.
REM
REM  Azure credentials come from the machine/user environment or the project's
REM  .env. They are never passed as arguments -- command lines are visible in
REM  the Windows process list and in UiPath job logs.
REM ===========================================================================

setlocal

REM --- Adjust these two for the deployment -------------------------------
set "PROJECT_DIR=%~dp0.."
set "PYTHON_EXE=%PROJECT_DIR%\env\Scripts\python.exe"
REM -----------------------------------------------------------------------

if "%~3"=="" (
    echo Usage: %~nx0 ^<ticket-file^> ^<history-file^> ^<output-dir^> 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    endlocal & exit /b 1
)

pushd "%PROJECT_DIR%"

REM stderr stays attached so faults land in the UiPath job log.
"%PYTHON_EXE%" -m scripts.run_resolver --ticket-file "%~1" --history-file "%~2" --output-dir "%~3"
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
