@echo off
setlocal enabledelayedexpansion

echo Parkinson's Disease Detection System
echo ===================================
echo.

:menu
echo Choose an option:
echo 1. Setup Environment
echo 2. Prepare Sample Data
echo 3. Train Models
echo 4. Evaluate Models
echo 5. Make Predictions (Web Interface)
echo 6. Run Tests
echo 7. Open Demo Notebook
echo 0. Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="1" (
    echo Setting up environment...
    python main.py setup
    echo.
    goto menu
)

if "%choice%"=="2" (
    echo Preparing sample data...
    python main.py prepare --create_sample --create_clinical --preprocess
    echo.
    goto menu
)

if "%choice%"=="3" (
    echo Training models...
    echo Choose model:
    echo 1. CNN
    echo 2. Random Forest
    echo 3. SVM
    echo 4. Gradient Boosting
    echo 5. Transfer Learning (VGG16)
    echo 6. Hybrid Model
    echo 7. All Models
    echo.
    
    set /p model_choice="Enter your choice (1-7): "
    
    if "!model_choice!"=="1" (
        python main.py train --model cnn
    ) else if "!model_choice!"=="2" (
        python main.py train --model rf
    ) else if "!model_choice!"=="3" (
        python main.py train --model svm
    ) else if "!model_choice!"=="4" (
        python main.py train --model gb
    ) else if "!model_choice!"=="5" (
        python main.py train --model transfer --transfer_model vgg16
    ) else if "!model_choice!"=="6" (
        python main.py train --model hybrid
    ) else if "!model_choice!"=="7" (
        python main.py train --train_all
    ) else (
        echo Invalid choice
    )
    
    echo.
    goto menu
)

if "%choice%"=="4" (
    echo Evaluating models...
    python main.py evaluate --plot --save_results
    echo.
    goto menu
)

if "%choice%"=="5" (
    echo Launching web interface for predictions...
    python main.py predict --web_interface
    echo.
    goto menu
)

if "%choice%"=="6" (
    echo Running tests...
    python main.py test
    echo.
    goto menu
)

if "%choice%"=="7" (
    echo Opening demo notebook...
    python main.py demo
    echo.
    goto menu
)

if "%choice%"=="0" (
    echo Exiting...
    exit /b 0
)

echo Invalid choice. Please try again.
echo.
goto menu
