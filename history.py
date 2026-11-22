"""Historique des discussions d'Athena (sauvegardé dans history.json, ignoré par Git).

- Une discussion = un titre (le début de la 1re question) + une liste d'échanges (question, réponse, sources).
- Les discussions NON épinglées sont supprimées quand elles n'ont pas eu d'activité depuis RETENTION_DAYS jours
  (3 par défaut). Les discussions épinglées sont gardées jusqu'au désépinglage.
- Chaque échange garde aussi ses « données d'affichage » (capsules de mails et d'agenda, cartes News IA, boutons) pour que la
  discussion rouverte ait la même mise en page qu'à l'origine.
- Un échange « privé » (mails, agenda) n'est conservé en entier que si garder_prive est vrai (réglage [historique] garder_prive de
  config.toml). Sinon il est enregistré SANS sa réponse ni ses capsules : seul un message neutre est gardé.
  Un échange privé n'est jamais redonné au modèle quand on continue la discussion.
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
MAX_EXTRA = 150000           # taille maximale (en caractères JSON) des données d'affichage gardées par échange
_ACTIONS = ("thunderbird", "agenda_connect")      # boutons d'action connus (liste blanche)
_EXTRA_DICTS = ("agenda", "mails", "veille")      # données d'affichage : capsules agenda, capsules mails, cartes News IA


def is_private(turn):
    """Échange « privé » (mails, agenda) : jamais redonné au modèle. Reconnaît aussi l'ancien format (note neutre à la place de la réponse)."""
    return bool(turn.get("private")) or turn.get("a") == PRIVATE_NOTE


def _clean_extra(extra, keep_private=True):
    """Données d'affichage d'un échange : copie propre (JSON simple, boutons de la liste blanche). Dictionnaire vide si rien d'utile.
    keep_private=False : les capsules de mails et d'agenda sont toujours retirées, même si on les reçoit par erreur."""
    out = {}
    if not isinstance(extra, dict): return out
    for key in _EXTRA_DICTS:
        val = extra.get(key)
        if not isinstance(val, dict): continue
        if key in ("agenda", "mails") and not keep_private: continue
        try: val = json.loads(json.dumps(val, ensure_ascii=False))       # copie profonde ; erreur si ce n'est pas du JSON simple
        except (TypeError, ValueError): continue
        out[key] = val
    actions = [a for a in (extra.get("actions") or []) if isinstance(a, str) and a in _ACTIONS]
    if actions: out["actions"] = list(dict.fromkeys(actions))
    if len(json.dumps(out, ensure_ascii=False)) > MAX_EXTRA: return {}      # trop gros : on garde seulement le texte
    return out


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
            turn = {"q": t["q"][:MAX_TEXT], "a": t["a"][:MAX_TEXT], "sources": _clean_sources(t.get("sources"))}
            if t.get("private") is True: turn["private"] = True
            extra = _clean_extra(t.get("extra"))
            if extra: turn["extra"] = extra
            if isinstance(t.get("part"), str) and t["part"].strip(): turn["part"] = t["part"][:300]
            turns.append(turn)
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

    def add_turn(self, cid, question, answer, sources=None, private=False, extra=None, part=None, keep_private=False):
        """Ajoute un échange (crée la discussion si `cid` est None ou inconnu). Retourne l'identifiant de la discussion.
        extra : données d'affichage {"agenda", "mails", "veille", "actions"} pour redessiner la réponse à l'identique.
        part : titre d'un bloc de la routine du matin (le titre fait partie du bloc, ce n'est pas une question de l'utilisateur).
        private + keep_private=False : seule la note neutre est gardée (ni réponse, ni liens, ni capsules)."""
        now = self._clock()
        c = self.get(cid) if cid else None
        if c is None:
            title = " ".join(question.split())[:60] or "Discussion"
            c = {"id": uuid.uuid4().hex[:12], "title": title, "created": now, "updated": now, "pinned": False, "turns": []}
            self.conversations.append(c)
        if private and not keep_private:
            turn = {"q": question[:MAX_TEXT], "a": PRIVATE_NOTE, "sources": []}
        else:
            turn = {"q": question[:MAX_TEXT], "a": answer[:MAX_TEXT], "sources": _clean_sources(sources)}
            if private: turn["private"] = True
            clean = _clean_extra(extra, keep_private=keep_private)
            if clean: turn["extra"] = clean
            if isinstance(part, str) and part.strip(): turn["part"] = part[:300]
        c["turns"].append(turn)
        c["turns"] = c["turns"][-MAX_TURNS:]
        c["updated"] = now
        self.purge(keep=c["id"])
        self.save()
        return c["id"]

    def scrub_private(self):
        """Efface de l'historique tout ce qui est privé (mails, agenda), pour les échanges déjà enregistrés : le texte devient la note
        neutre et les capsules disparaissent. À appeler quand garder_prive est désactivé. Retourne le nombre d'échanges modifiés."""
        changed = 0
        for c in self.conversations:
            for t in c["turns"]:
                if t.get("private"):
                    t["a"], t["sources"] = PRIVATE_NOTE, []
                    for k in ("private", "extra", "part"): t.pop(k, None)
                    changed += 1
                elif isinstance(t.get("extra"), dict) and ("agenda" in t["extra"] or "mails" in t["extra"]):
                    t["extra"] = {k: v for k, v in t["extra"].items() if k not in ("agenda", "mails")}
                    if not t["extra"]: del t["extra"]
                    changed += 1
        if changed: self.save()
        return changed

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
