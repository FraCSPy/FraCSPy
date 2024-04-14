:: Windows installer script for pyfrac
:: This script creates a Conda environment based on an "environment.yml" file and installs pyfrac
:: Run: install_pyfrac.bat from cmd
:: D. Anikiev, 03/04/2024

@echo off

set ENV_YAML=environment.yml
set ENV_NAME=pyfrac

:: Check for Conda Installation
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not found in the system PATH.
    echo Please install Conda and make sure it's added to the system PATH.
    pause
    exit /b 1
)

:: Check for environment file
if not exist %ENV_YAML% (
    echo %ENV_YAML% not found in the current directory
    echo Please check and try again.
    pause
    exit /b 2 
)

echo Creating Conda environment %ENV_NAME% from %ENV_YAML%...
call conda env create -f %ENV_YAML%
:: Check if environment creation was successful
if %ERRORLEVEL% equ 0 (
    echo Conda environment %ENV_NAME% created successfully.
) else (
    echo Failed to create Conda environment %ENV_NAME%. Please check the error messages above.
    pause
    exit /b 3
)

:: List conda environments
call conda env list

:: Activate environment
echo Activating Conda environment %ENV_NAME%...
call activate %ENV_NAME%
:: Check if environment activation was successful
if %ERRORLEVEL% equ 0 (    
    echo Conda environment %ENV_NAME% activated successfully.
) else (
    echo Failed to activate Conda environment %ENV_NAME%. Please check the error messages above.
    exit /b 4
    pause
)

echo Installing pyfrac...
call pip install -e .
if %ERRORLEVEL% equ 0 (    
    echo Successfully installed pyfrac.
) else (
    echo Failed to installed pyfrac. Please check the error messages above.
    pause
    exit /b 5
)

echo Python version:
call python --version

echo Python path:
:: Pick only first output
for /f "tokens=* usebackq" %%f in (`where python`) do (set "pythonpath=%%f" & goto :next)
:next

echo %pythonpath%

echo Done!

pause
