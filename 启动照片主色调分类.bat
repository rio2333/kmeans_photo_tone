@echo off
chcp 65001 > nul
cd /d "%~dp0"
python photo_tone_classifier.py
echo.
pause
