"""Connecte Athena à ton Google Agenda (LECTURE SEULE), une seule fois. Lance-le avec Connecter-agenda.bat.

Avant : crée les variables d'environnement ATHENA_GOOGLE_CLIENT_ID et ATHENA_GOOGLE_CLIENT_SECRET (voir le README, section « Agenda Google »).
Ce script ouvre ton navigateur ; tu te connectes à Google et tu acceptes. La clé de rafraîchissement est enregistrée dans agenda_token.json
(à côté d'Athena, ignoré par Git). Pour retirer l'accès : compte Google > Sécurité > Applications tierces > Athena > Supprimer l'accès.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import agenda
import engine

print("=== Connexion d'Athena à Google Agenda (lecture seule) ===")
client_id = engine.get_secret("ATHENA_GOOGLE_CLIENT_ID")
client_secret = engine.get_secret("ATHENA_GOOGLE_CLIENT_SECRET")
try:
    n = agenda.connect(client_id, client_secret, print)
    print(f"\nConnecté. Athena voit {n} agenda(s) sur ton compte. Tu peux lui demander « j'ai quoi de prévu aujourd'hui ? ».")
except agenda.AgendaError as e:
    print(f"\nÉchec : {e}")
    sys.exit(1)
