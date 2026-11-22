"""Lance Athena SANS fenêtre de console.

Utilisation : raccourci « Athena » (Athena.lnk, qui appelle pythonw.exe du venv avec ce fichier) ou clic droit > Ouvrir avec pythonw.
Les messages qui s'affichaient dans la console vont dans le fichier athena.log (à côté de ce fichier, ignoré par Git) :
ouvre-le si Athena ne démarre pas ou plante. Pour voir la console en direct, utilise Athena-console.bat.
"""
import os
import sys
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# Un double-clic direct sur ce fichier utilise le Python « système » de Windows, qui n'a pas Flet : on se relance alors avec celui du venv.
VENV_PYTHONW = os.path.join(os.path.dirname(HERE), "venv", "Scripts", "pythonw.exe")      # D:\Athena\venv (à côté du dossier « app »)
if (os.path.isfile(VENV_PYTHONW)
        and os.path.normcase(os.path.dirname(os.path.abspath(sys.executable))) != os.path.normcase(os.path.dirname(VENV_PYTHONW))):
    subprocess.Popen([VENV_PYTHONW, os.path.abspath(__file__)] + sys.argv[1:], cwd=HERE,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    sys.exit(0)

# Sans console, print() et les erreurs n'iraient nulle part : on les écrit dans athena.log (1 Mo max, puis un seul ancien fichier gardé)
LOG = os.path.join(HERE, "athena.log")
try:
    if os.path.exists(LOG) and os.path.getsize(LOG) > 1_000_000:
        os.replace(LOG, LOG + ".old")
    sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
    print(f"\n===== Lancement d'Athena : {datetime.datetime.now():%d/%m/%Y %H:%M:%S} =====")
except OSError:
    pass

import logs
logs.install()                        # chaque ligne du journal commence par l'heure : [16:34:11] ...

import flet as ft
import ui

if "--verifier" in sys.argv:          # vérification sans ouvrir la fenêtre : « python Athena.pyw --verifier »
    print("[Lanceur] Imports OK (flet, ui, engine).")
    sys.exit(0)

ft.run(ui.main)
