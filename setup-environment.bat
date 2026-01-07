@echo off
REM YouTube-SC project environment setup script (Windows)
REM Use uv to manage Python environment

echo ========================================
echo    YouTube-SC Project Environment Setup
echo ========================================
echo.

REM Check Python
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

REM Check uv Install
echo.
echo Check uv Install...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo uv 未Install，正在Install...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    REM 重新Check
    where uv >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: uv Install失败
        echo PleasemanuallyInstall: https://github.com/astral-sh/uv
        pause
        exit /b 1
    )
    echo ✅ uv Install成功
) else (
    echo ✅ uv 已Install
)

REM Createvirtual environment
echo.
echo Createvirtual environment...
uv venv --python 3.11 .venv
if %errorlevel% neq 0 (
    echo Error: Createvirtual environment失败
    pause
    exit /b 1
)
echo ✅ virtual environmentCreate成功

REM Activate环境
echo.
echo Activatevirtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Activatevirtual environment失败
    pause
    exit /b 1
)
echo ✅ virtual environment已Activate

REM Installdependencies
echo.
echo Installprojectdependencies...
uv sync
if %errorlevel% neq 0 (
    echo Warning: dependenciesInstall遇到问题
    echo tryalternative...
    
    REM try从 requirements.txt Install
    if exist requirements.txt (
        echo 从 requirements.txt Install...
        uv add -r requirements.txt
    ) else (
        echo Error: not founddependenciesfile
        echo PleasemanuallyInstalldependencies
    )
)

REM VerifyInstall
echo.
echo VerifyInstall...
python -c "import pandas; print(f'✅ pandas version: {pandas.__version__}')"
python -c "import sklearn; print(f'✅ scikit-learn version: {sklearn.__version__}')"
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" 2>nul || echo "⚠ PyTorch 未Install（optional）"

echo.
echo ========================================
echo    Environment setup completed!
echo ========================================
echo.
echo Available commands:
echo   - Runclustering analysis: cd Code\sdr_clustering_analysis && python main.py
echo   - RunML sentiment classification: cd Code\sentiment_classification_ML && python main.py --help
echo   - RunBERT sentiment classification: cd Code\sentiment_classification_Bert\code && python main.py --help
echo   - Runtopic modeling: cd Code\topic_modeling && python topic_modeling_analysis.py
echo.
echo Next time, activate the environment first:
echo   .venv\Scripts\activate.bat
echo.
pause