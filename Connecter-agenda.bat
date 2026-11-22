@echo off
rem Connecte Athena a Google Agenda (lecture seule), une seule fois. Ouvre ton navigateur : connecte-toi a Google et accepte.
cd /d "%~dp0"
"%~dp0..\venv\Scripts\python.exe" connecter_agenda.py
echo.
pause
