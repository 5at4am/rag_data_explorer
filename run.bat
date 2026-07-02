@echo off
chcp 65001 >nul
echo Installing dependencies...
pip install -r requirements.txt
echo Starting app...
streamlit run app/main.py
pause
