@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "JAVA_PATH=%SCRIPT_DIR%jdk-17\bin\java.exe"

if not exist "%JAVA_PATH%" (
    echo Could not find bundled Java at:
    echo %JAVA_PATH%
    pause
    exit /b 1
)

"%JAVA_PATH%" -jar "%SCRIPT_DIR%bayes.jar"
pause
