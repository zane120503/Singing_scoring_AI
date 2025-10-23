@echo off
echo ========================================
echo    Hệ thống chấm điểm karaoke bằng AI
echo ========================================
echo.

echo Đang kiểm tra Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python chưa được cài đặt!
    echo Vui lòng cài đặt Python 3.8+ từ https://python.org
    pause
    exit /b 1
)

echo ✅ Python đã được cài đặt
echo.

echo Đang cài đặt dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Lỗi khi cài đặt dependencies!
    pause
    exit /b 1
)

echo ✅ Dependencies đã được cài đặt
echo.

echo Đang khởi động hệ thống...
python main.py

pause

