@echo off
setlocal EnableDelayedExpansion
title SC Analysis - installer
cd /d "%~dp0"
set ENV=sc_analysis

echo.
echo === SC Analysis installer ===
echo.

rem ---- find conda (Miniconda / Anaconda) ----
set "CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA=%CONDA_EXE%"
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\Anaconda3" "C:\miniconda3" "C:\Anaconda3") do (
  if not defined CONDA if exist "%%~D\Scripts\conda.exe" set "CONDA=%%~D\Scripts\conda.exe"
)
if not defined CONDA (
  for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
if not defined CONDA (
  echo Could not find conda. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
  echo and run this installer again.
  pause
  exit /b 1
)
echo Using conda: %CONDA%

rem ---- create / update the environment ----
call "%CONDA%" env list | findstr /b /c:"%ENV% " >nul
if errorlevel 1 (
  echo Creating environment %ENV% ^(Python 3.11^)...
  call "%CONDA%" create -y -n %ENV% --override-channels -c conda-forge python=3.11
  if errorlevel 1 goto :fail
) else (
  echo Environment %ENV% already exists - updating it.
)

for /f "delims=" %%P in ('call "%CONDA%" run -n %ENV% python -c "import sys; print(sys.prefix)"') do set "PREFIX=%%P"
for %%C in ("%CONDA%") do set "CROOT=%%~dpC.."
if not exist "%PREFIX%\python.exe" set "PREFIX=%CROOT%\envs\%ENV%"
if not exist "%PREFIX%\python.exe" goto :fail
echo Environment: %PREFIX%

"%PREFIX%\python.exe" -m pip install --upgrade pip
"%PREFIX%\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail
"%PREFIX%\python.exe" -m pip install --no-deps --upgrade "%~dp0."
if errorlevel 1 goto :fail

rem ---- launcher + desktop shortcut ----
set "APPDIR=%LOCALAPPDATA%\SCAnalysis"
if not exist "%APPDIR%" mkdir "%APPDIR%"
(
  echo @echo off
  echo start "" "%PREFIX%\pythonw.exe" -m sc_analysis %%*
) > "%APPDIR%\SC Analysis.bat"
(
  echo @echo off
  echo "%PREFIX%\python.exe" -m sc_analysis %%*
  echo pause
) > "%APPDIR%\SC Analysis (console).bat"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\SC Analysis.lnk');" ^
  "$s.TargetPath='%PREFIX%\pythonw.exe'; $s.Arguments='-m sc_analysis'; $s.WorkingDirectory='%USERPROFILE%';" ^
  "$s.Description='SC Neuropixels analysis'; $s.IconLocation='%PREFIX%\Lib\site-packages\sc_analysis\resources\icon.ico,0'; $s.Save()"

echo.
echo Done. Start "SC Analysis" from the desktop shortcut
echo (or "%APPDIR%\SC Analysis (console).bat" to see error messages).
echo.
pause
exit /b 0

:fail
echo.
echo Installation failed - see the messages above.
pause
exit /b 1
