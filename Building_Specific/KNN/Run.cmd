@echo off
set "ROOT=%~dp0..\.."
cd /d "%ROOT%"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (echo Run Setup.cmd first. & pause & exit /b 1)
:menu
cls
echo Building_Specific - KNN
echo 1. Train: grouped CV and full-training refit
echo 2. Evaluate: official validation benchmark
echo 3. Demo: three TRAINING rehearsal scans
echo 4. Open comparison and PCA/k plots
echo 5. Exit
choice /c 12345 /n /m "Choose [1-5]: "
if errorlevel 5 exit /b 0
if errorlevel 4 goto report
if errorlevel 3 goto demo
if errorlevel 2 goto evaluate
if errorlevel 1 goto train
:train
if not exist "Building_Specific\KNN\Saved_Models\KNN_Metadata.json" goto firsttrain
echo This replaces saved models and invalidates their old evaluation.
choice /c YN /n /m "Replace models? [Y/N] "
if errorlevel 2 goto menu
"%PY%" "Building_Specific\KNN\model.py" train --overwrite
goto done
:firsttrain
"%PY%" "Building_Specific\KNN\model.py" train
goto done
:evaluate
"%PY%" "Building_Specific\KNN\model.py" evaluate
goto done
:demo
"%PY%" "Building_Specific\KNN\model.py" demo
goto done
:report
if not exist "Results\Model_Comparison.html" (echo Train/evaluate first. & goto done)
start "" "Results\Model_Comparison.html"
:done
if errorlevel 1 echo The action failed. Check the error above.
pause
goto menu
