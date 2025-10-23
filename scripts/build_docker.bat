@echo off
echo ========================================
echo    Building Karaoke Scoring Docker
echo ========================================
echo.

echo Building Docker image...
docker-compose build

if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo ✅ Build completed successfully!
echo.
echo To run the application:
echo docker-compose up
echo.
pause

