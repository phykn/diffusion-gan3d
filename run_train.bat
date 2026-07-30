@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [DIFFUSION-GAN3D] .venv\Scripts\python.exe was not found.
    echo [DIFFUSION-GAN3D] Create the virtual environment and install requirements first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_train.py %*
set "DIFFUSION_GAN3D_EXIT_CODE=%ERRORLEVEL%"

if not "%DIFFUSION_GAN3D_EXIT_CODE%"=="0" (
    echo [DIFFUSION-GAN3D] Training exited with code %DIFFUSION_GAN3D_EXIT_CODE%.
)

pause
exit /b %DIFFUSION_GAN3D_EXIT_CODE%
