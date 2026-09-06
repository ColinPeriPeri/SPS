@echo off
REM ===========================================================================
REM  SPS inference wrapper for the UiPath Performer.
REM
REM    run_inference.cmd <payload-file> <output-file.xlsx> <status-file>
REM
REM  Exit codes (propagated verbatim to UiPath):
REM    0  pipeline ran. Includes "Solution not found." -- a business outcome.
REM    1  infrastructure fault. Raise a system exception and retry the item.
REM    2  the payload file was not valid JSON. Fault the item, do not retry.
REM
REM  <output-file.xlsx> is a workbook: one row per ticket, four contract
REM  columns. Any other extension produces JSON text instead.
REM
REM  <status-file> is three lines, written LAST and only once the output file
REM  is complete:
REM      STATUS: SUCCESS | FAILURE
REM      EXIT_CODE: 0 | 1 | 2
REM      REASON: <one-line explanation>
REM  STATUS is SUCCESS if and only if EXIT_CODE is 0, so the two can never
REM  disagree. A SUCCESS status guarantees the workbook beside it is readable.
REM
REM  If this process is killed outright neither file is written, so "status
REM  file missing" and "non-zero exit" both mean the transaction is
REM  untrustworthy. Read the workbook only after the status file says SUCCESS.
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
    echo Usage: %~nx0 ^<payload-file^> ^<output-file.xlsx^> ^<status-file^> 1>&2
    endlocal & exit /b 2
)

if not exist "%PYTHON_EXE%" (
    echo Python interpreter not found: "%PYTHON_EXE%" 1>&2
    endlocal & exit /b 1
)

pushd "%PROJECT_DIR%"

REM stdout is swallowed: the contract is read from the output file. stderr is
REM left attached so faults land in the UiPath job log.
"%PYTHON_EXE%" -m service.run_inference --payload-file "%~1" --output-file "%~2" --status-file "%~3" >nul
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
