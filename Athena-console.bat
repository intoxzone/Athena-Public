@echo off
rem Lance Athena AVEC la console (pour voir les messages en direct et les erreurs). Pour un lancement normal, utilise le raccourci Athena.
cd /d "%~dp0"
"%~dp0..\venv\Scripts\python.exe" ui.py
if errorlevel 1 (
    echo.
    echo Athena s'est arretee avec une erreur. Copie le texte ci-dessus pour le montrer.
    pause
)
