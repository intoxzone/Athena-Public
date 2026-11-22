"""Historique des discussions d'Athena (sauvegardé dans history.json, ignoré par Git).

- Une discussion = un titre (le début de la 1re question) + une liste d'échanges (question, réponse, sources).
- Les discussions NON épinglées sont supprimées quand elles n'ont pas eu d'activité depuis RETENTION_DAYS jours
  (3 par défaut). Les discussions épinglées sont gardées jusqu'au désépinglage.
- Un échange « privé » (résumé de mails) est enregistré SANS sa réponse : seul un message neutre est gardé.
- Le fichier est du texte clair sur ton PC : ne le partage pas. Il est écrit de façon atomique (fichier temporaire puis remplacement).
"""
import json
import os
import time
import uuid

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")
RETENTION_DAYS = 3
MAX_TURNS = 100              # échanges gardés par discussion
MAX_CONVERSATIONS = 80       # discussions gardées au total (les plus anciennes non épinglées partent en premier)
MAX_TEXT = 20000             # caractères gardés par question ou réponse
PRIVATE_NOTE = "(Le résumé des mails n'est pas conservé dans l'historique.)"
_SOURCE_KEYS = ("title", "url", "kind", "subtitle", "snippet")


def _clean_sources(sources):
    out = []
    for s in sources or []:
        if not isinstance(s, dict) or not str(s.get("url", "")).lower().startswith(("http://", "https://")): continue
        out.append({k: str(s[k])[:500] for k in _SOURCE_KEYS if s.get(k)})
    return out[:12]


def _clean_conversation(c):
    """Conversation lue dans le fichier : retourne un dict propre, ou None si elle est inutilisable."""
    if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("turns"), list): return None
    turns = []
    for t in c["turns"]:
        if isinstance(t, dict) and isinstance(t.get("q"), str) and isinstance(t.get("a"), str):
            turns.append({"q": t["q"][:MAX_TEXT], "a": t["a"][:MAX_TEXT], "sources": _clean_sources(t.get("sources"))})
    if not turns: return None
    try: created, updated = float(c.get("created", 0)), float(c.get("updated", 0))
    except (TypeError, ValueError): return None
    return {"id": c["id"], "title": str(c.get("title") or turns[0]["q"])[:80], "created": created, "updated": updated,
            "pinned": bool(c.get("pinned")), "turns": turns[-MAX_TURNS:]}


class HistoryStore:
    def __init__(self, path=None, retention_days=None, clock=time.time):
        self.path = path or HISTORY_PATH
        self.retention = (RETENTION_DAYS if retention_days is None else retention_days) * 86400
        self._clock = clock
        self.conversations = []
        self.load()

    # ------------------------------------------------------------------ lecture / écriture
    def load(self):
        self.conversations = []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("conversations", []) if isinstance(data, dict) else []
            self.conversations = [c for c in (_clean_conversation(x) for x in raw) if c]
        except FileNotFoundError:
            pass
        except (OSError, ValueError):                      # fichier abîmé : on le met de côté au lieu de le perdre ou de planter
            try: os.replace(self.path, self.path + ".bad")
            except OSError: pass
        if self.purge(): self.save()

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "conversations": self.conversations}, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
            return True
        except OSError as e:
            print(f"[Historique] enregistrement impossible : {e}")
            return False

    # ------------------------------------------------------------------ règles de conservation
    def purge(self, keep=None):
        """Supprime les discussions non épinglées sans activité depuis la durée de conservation
        (sauf la discussion `keep`, celle qui est ouverte). Retourne True si quelque chose a été supprimé."""
        limit = self._clock() - self.retention
        before = len(self.conversations)
        self.conversations = [c for c in self.conversations if c["pinned"] or c["id"] == keep or c["updated"] >= limit]
        if len(self.conversations) > MAX_CONVERSATIONS:
            loose = sorted((c for c in self.conversations if not c["pinned"] and c["id"] != keep), key=lambda c: c["updated"])
            for c in loose[:len(self.conversations) - MAX_CONVERSATIONS]:
                self.conversations.remove(c)
        return len(self.conversations) != before

    # ------------------------------------------------------------------ utilisation
    def get(self, cid):
        return next((c for c in self.conversations if c["id"] == cid), None)

    def add_turn(self, cid, question, answer, sources=None, private=False):
        """Ajoute un échange (crée la discussion si `cid` est None ou inconnu). Retourne l'identifiant de la discussion."""
        now = self._clock()
        c = self.get(cid) if cid else None
        if c is None:
            title = " ".join(question.split())[:60] or "Discussion"
            c = {"id": uuid.uuid4().hex[:12], "title": title, "created": now, "updated": now, "pinned": False, "turns": []}
            self.conversations.append(c)
        c["turns"].append({"q": question[:MAX_TEXT], "a": PRIVATE_NOTE if private else answer[:MAX_TEXT],
                           "sources": [] if private else _clean_sources(sources)})
        c["turns"] = c["turns"][-MAX_TURNS:]
        c["updated"] = now
        self.purge(keep=c["id"])
        self.save()
        return c["id"]

    def pin(self, cid, value=True):
        c = self.get(cid)
        if c is None: return False
        c["pinned"] = bool(value)
        self.save()
        return True

    def delete(self, cid):
        n = len(self.conversations)
        self.conversations = [c for c in self.conversations if c["id"] != cid]
        if len(self.conversations) != n: self.save()
        return len(self.conversations) != n

    def items(self, keep=None):
        """Liste pour la barre latérale : les épinglées d'abord, puis les autres, chacune de la plus récente à la plus ancienne."""
        if self.purge(keep=keep): self.save()
        order = sorted(self.conversations, key=lambda c: (not c["pinned"], -c["updated"]))
        return [{"id": c["id"], "title": c["title"], "pinned": c["pinned"], "updated": c["updated"], "turns": len(c["turns"])} for c in order]
