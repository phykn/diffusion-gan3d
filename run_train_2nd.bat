@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [DIFFUSION-GAN3D] Create .venv and install requirements first.
    pause
    exit /b 1
)

if not "%~1"=="" goto arguments
echo Enter the stage-1 run folder or generator.pt path.
set /p "DIFFUSION_GAN3D_BASE=Stage-1 weights: "
if not defined DIFFUSION_GAN3D_BASE exit /b 1
set "DIFFUSION_GAN3D_BASE=%DIFFUSION_GAN3D_BASE:"=%"
".venv\Scripts\python.exe" run_train_2nd.py --base-weights "%DIFFUSION_GAN3D_BASE%"
goto finished

:arguments
".venv\Scripts\python.exe" run_train_2nd.py %*

:finished
set "DIFFUSION_GAN3D_EXIT_CODE=%ERRORLEVEL%"
if not "%DIFFUSION_GAN3D_EXIT_CODE%"=="0" (
    echo [DIFFUSION-GAN3D] Training exited with code %DIFFUSION_GAN3D_EXIT_CODE%.
)
pause
exit /b %DIFFUSION_GAN3D_EXIT_CODE%
