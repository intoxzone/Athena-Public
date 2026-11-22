"""Horodatage des messages de la console (et du fichier athena.log) : chaque ligne commence par [HH:MM:SS].

install() enveloppe sys.stdout et sys.stderr : tout ce qui est écrit (print d'Athena, messages [Config], erreurs Python)
reçoit l'heure en début de ligne. Les lignes vides n'en reçoivent pas. Sans effet si déjà installé ou s'il n'y a pas de console.
"""
import sys
import threading
import time


def _stamp():
    return time.strftime("[%H:%M:%S] ")


class _Stamped:
    """Flux qui ajoute l'heure au début de chaque ligne écrite (même quand une ligne arrive en plusieurs morceaux)."""

    def __init__(self, stream):
        self._stream = stream
        self._at_start = True
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        with self._lock:
            out = []
            for i, line in enumerate(text.split("\n")):
                if i > 0:
                    out.append("\n")
                    self._at_start = True
                if line:
                    if self._at_start:
                        out.append(_stamp())
                        self._at_start = False
                    out.append(line)
            self._stream.write("".join(out))
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):                 # encoding, isatty, fileno... : on laisse le flux d'origine répondre
        return getattr(self._stream, name)


def install():
    """Active l'horodatage sur la sortie standard et la sortie d'erreur (une seule fois)."""
    if isinstance(sys.stdout, _Stamped) or sys.stdout is None:
        return
    out, err = sys.stdout, sys.stderr
    stamped = _Stamped(out)
    sys.stdout = stamped
    if err is out:
        sys.stderr = stamped                     # même fichier : un seul flux pour que l'ordre des lignes soit respecté
    elif err is not None:
        sys.stderr = _Stamped(err)
