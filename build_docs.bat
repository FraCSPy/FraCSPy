:: Windows script for building pyfrac documentation
:: This script uses docs/Makefile to build the sphinx documentation
:: Run: build_docs.bat from cmd
:: D. Anikiev, 24/04/2024

@echo off

cd docs

:: Check if build exists
if exist build (
    echo Cleaning...
    call make clean
    :: Check
    if %ERRORLEVEL% neq 0 (
        echo Error while cleaning!
        exit /b 1
    )
)

:: Make documentation
echo Building HTML...
call make html
:: Check
if %ERRORLEVEL% neq 0 (
    echo Error during build process!
    exit /b 1
)
  
:: Serve
echo Starting server (please check the server port)...
cd build\html
python -m http.server
:: Check
if %ERRORLEVEL% neq 0 (
    echo Error while running server!
    exit /b 3
)

cd ..\..\..
echo Done!
