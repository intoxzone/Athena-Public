"""Consultation de Google Agenda (LECTURE SEULE) : « j'ai quoi de prévu aujourd'hui / demain / cette semaine ».

- Connexion OAuth « application de bureau » avec le droit calendar.readonly : Athena ne peut ni créer, ni modifier, ni supprimer un événement.
  Une seule fois : Connecter-agenda.bat (navigateur, tu acceptes) ; ensuite l'autorisation se rafraîchit toute seule.
- La clé de rafraîchissement est dans agenda_token.json (à côté d'Athena, ignoré par Git). Les clés Google (identifiant client et clé secrète)
  sont dans les variables d'environnement ATHENA_GOOGLE_CLIENT_ID et ATHENA_GOOGLE_CLIENT_SECRET, jamais dans un fichier du projet.
- Google calcule lui-même les rendez-vous qui se répètent (singleEvents=true) : aucune bibliothèque de calendrier n'est nécessaire.
- Rien n'est donné à Qwen, rien n'est affiché dans les logs : seulement des nombres. Le texte est fabriqué par le code.
Ce module ne dépend pas d'engine.py (c'est engine.py qui l'utilise).
"""
import base64
import datetime
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import unicodedata
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import requests

TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agenda_token.json")
SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/calendar/v3"
TIMEOUT = 15.0
MAX_EVENTS = 80                                 # événements affichés au maximum (les suivants sont comptés, pas listés)

JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre")


class AgendaError(Exception):
    """Lecture de l'agenda impossible : le message (en français, sans clé ni jeton) est affiché tel quel.
    reconnect=True : l'accès est à renouveler (jamais connecté, autorisation expirée) → l'interface propose le bouton qui lance Connecter-agenda.bat."""
    def __init__(self, message, reconnect=False):
        super().__init__(message)
        self.reconnect = reconnect


# =====================================================================================
#  Autorisation Google (OAuth, jeton de rafraîchissement dans agenda_token.json)
# =====================================================================================
_access = {"token": None, "expires": 0.0}


def _load_token():
    try:
        with open(TOKEN_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data.get("refresh_token") else None
    except (OSError, ValueError):
        return None


def _save_token(data):
    tmp = TOKEN_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, TOKEN_PATH)


def is_connected():
    return _load_token() is not None


def access_token(client_id, client_secret):
    """Jeton d'accès valide (rafraîchi si besoin). Lève AgendaError avec la marche à suivre."""
    if _access["token"] and time.time() < _access["expires"]:
        return _access["token"]
    if not client_id or not client_secret:
        raise AgendaError("les clés Google ne sont pas définies : crée les variables ATHENA_GOOGLE_CLIENT_ID et ATHENA_GOOGLE_CLIENT_SECRET "
                          "(setx dans PowerShell), puis lance Connecter-agenda.bat")
    tok = _load_token()
    if not tok:
        raise AgendaError("Athena n'est pas encore connectée à Google Agenda : lance Connecter-agenda.bat (une seule fois)", reconnect=True)
    try:
        r = requests.post(TOKEN_URL, timeout=TIMEOUT, data={"client_id": client_id, "client_secret": client_secret,
                                                             "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"})
    except requests.exceptions.RequestException:
        raise AgendaError("Google est injoignable (connexion internet ?)")
    if r.status_code != 200:
        try: err = r.json().get("error", "")
        except ValueError: err = ""
        if err == "invalid_grant":
            raise AgendaError("l'autorisation de Google a expiré ou a été retirée : relance Connecter-agenda.bat. Si ça revient chaque semaine, "
                              "passe ton application « en production » dans la console Google Cloud (écran de consentement OAuth)", reconnect=True)
        if err in ("invalid_client", "unauthorized_client"):
            raise AgendaError("Google refuse les clés (identifiant ou clé secrète faux) : vérifie ATHENA_GOOGLE_CLIENT_ID et ATHENA_GOOGLE_CLIENT_SECRET")
        raise AgendaError(f"Google a refusé la connexion (code {r.status_code})")
    try:
        data = r.json()
        _access["token"] = data["access_token"]
        _access["expires"] = time.time() + max(60, int(data.get("expires_in", 3600)) - 120)
    except (ValueError, KeyError):
        raise AgendaError("réponse inattendue de Google")
    return _access["token"]


def connect(client_id, client_secret, log=print, open_browser=True, timeout=240):
    """Connexion unique : ouvre le navigateur, tu acceptes la lecture de tes agendas, la clé de rafraîchissement est enregistrée.
    Retourne le nombre d'agendas trouvés. Le serveur local n'écoute que sur 127.0.0.1 et vérifie le paramètre « state »."""
    if not client_id or not client_secret:
        raise AgendaError("les clés Google ne sont pas définies (ATHENA_GOOGLE_CLIENT_ID et ATHENA_GOOGLE_CLIENT_SECRET)")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    got = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in q or "error" in q:
                got.update({k: v[0] for k, v in q.items()})
            ok = "code" in q and q.get("state", [""])[0] == state
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(("<html><body style='font-family:sans-serif;padding:40px'><h2>%s</h2></body></html>" % (
                "Athena est connectée à Google Agenda. Tu peux fermer cette page." if ok else "Connexion refusée ou invalide. Tu peux fermer cette page.")).encode("utf-8"))
        def log_message(self, *a): pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{server.server_address[1]}"
    url = AUTH_URL + "?" + urllib.parse.urlencode({"client_id": client_id, "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
                                                  "access_type": "offline", "prompt": "consent", "state": state,
                                                  "code_challenge": challenge, "code_challenge_method": "S256"})
    log("Ouverture de ton navigateur : connecte-toi à Google et accepte (lecture seule de tes agendas).")
    log("Si un avertissement « Google n'a pas validé cette application » apparaît : Paramètres avancés, puis continuer (c'est ton application).")
    if open_browser: webbrowser.open(url)
    server.timeout = 1.0
    deadline = time.time() + timeout
    while time.time() < deadline and "code" not in got and "error" not in got:
        server.handle_request()
    server.server_close()
    if "error" in got: raise AgendaError(f"connexion refusée dans le navigateur ({got['error']})")
    if "code" not in got: raise AgendaError("pas de réponse du navigateur dans le temps imparti : relance Connecter-agenda.bat")
    if got.get("state") != state: raise AgendaError("réponse du navigateur invalide (état différent) : relance Connecter-agenda.bat")
    try:
        r = requests.post(TOKEN_URL, timeout=TIMEOUT, data={"client_id": client_id, "client_secret": client_secret, "code": got["code"], "redirect_uri": redirect,
                                                             "grant_type": "authorization_code", "code_verifier": verifier})
    except requests.exceptions.RequestException:
        raise AgendaError("Google est injoignable (connexion internet ?)")
    if r.status_code != 200:
        raise AgendaError("Google a refusé le code de connexion : vérifie l'identifiant et la clé secrète, puis relance Connecter-agenda.bat")
    data = r.json()
    if not data.get("refresh_token"):
        raise AgendaError("Google n'a pas donné de clé de rafraîchissement : retire l'accès d'Athena dans ton compte Google (Sécurité > Applications "
                          "tierces), puis relance Connecter-agenda.bat")
    _save_token({"refresh_token": data["refresh_token"], "scope": data.get("scope", SCOPE), "created": int(time.time())})
    _access["token"], _access["expires"] = data.get("access_token"), time.time() + max(60, int(data.get("expires_in", 3600)) - 120)
    return len(list_calendars(client_id, client_secret))


# =====================================================================================
#  Lecture des agendas et des événements
# =====================================================================================
def _norm(text):
    text = "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _api_get(path, params, client_id, client_secret):
    for attempt in (1, 2):
        token = access_token(client_id, client_secret)
        try:
            r = requests.get(API + path, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT)
        except requests.exceptions.RequestException:
            raise AgendaError("Google est injoignable (connexion internet ?)")
        if r.status_code == 401 and attempt == 1:
            _access["token"] = None                                  # jeton refusé : on en redemande un
            continue
        break
    if r.status_code == 403:
        try: reason = json.dumps(r.json())[:300]
        except ValueError: reason = ""
        if "accessNotConfigured" in reason or "has not been used" in reason or "SERVICE_DISABLED" in reason:
            raise AgendaError("l'API « Google Calendar » n'est pas activée dans ton projet Google Cloud (console.cloud.google.com > API et services)")
        raise AgendaError("accès refusé par Google (droits insuffisants sur cet agenda)")
    if r.status_code == 429: raise AgendaError("trop de demandes envoyées à Google, réessaie dans un instant")
    if r.status_code == 404: raise AgendaError("agenda introuvable")
    if r.status_code != 200: raise AgendaError(f"Google a répondu une erreur (code {r.status_code})")
    try: return r.json()
    except ValueError: raise AgendaError("réponse illisible de Google")


def list_calendars(client_id, client_secret, exclude=()):
    """[{id, name}] de TOUS les agendas du compte (perso, partagés, jours fériés...), hors ceux masqués ou exclus par nom (réglage [agenda] exclure)."""
    skip = {_norm(x) for x in exclude}
    out, page = [], None
    while True:
        params = {"fields": "nextPageToken,items(id,summary,summaryOverride,hidden,deleted,accessRole)", "maxResults": 250}
        if page: params["pageToken"] = page
        data = _api_get("/users/me/calendarList", params, client_id, client_secret)
        for c in data.get("items") or []:
            if c.get("hidden") or c.get("deleted") or not c.get("id"): continue
            name = " ".join(str(c.get("summaryOverride") or c.get("summary") or c["id"]).split())
            if _norm(name) in skip: continue
            out.append({"id": c["id"], "name": name, "mine": c.get("accessRole") == "owner"})       # « owner » = ton agenda ; sinon partagé avec toi
        page = data.get("nextPageToken")
        if not page: return out


def _parse_event(item, calendar, mine=True):
    """Événement Google → dict, ou None (annulé, refusé par toi, lieu de travail, illisible)."""
    if item.get("status") == "cancelled" or item.get("eventType") == "workingLocation": return None
    if any(a.get("self") and a.get("responseStatus") == "declined" for a in item.get("attendees") or []): return None
    s, e = item.get("start") or {}, item.get("end") or {}
    try:
        if "date" in s:
            start = datetime.date.fromisoformat(s["date"]); end = datetime.date.fromisoformat(e.get("date") or s["date"]); all_day = True
        elif "dateTime" in s:
            start = datetime.datetime.fromisoformat(s["dateTime"]).astimezone()
            end = datetime.datetime.fromisoformat(e["dateTime"]).astimezone() if e.get("dateTime") else start; all_day = False
        else:
            return None
    except (ValueError, TypeError):
        return None
    return {"start": start, "end": end, "all_day": all_day, "calendar": calendar, "mine": bool(mine),
            "title": " ".join(str(item.get("summary") or "(sans titre)").split())[:120], "location": " ".join(str(item.get("location") or "").split())[:80]}


def _local_midnight(d):
    return datetime.datetime.combine(d, datetime.time.min).astimezone()


def get_events(first, last, client_id, client_secret, exclude=()):
    """Événements de tous les agendas du jour `first` au jour `last` inclus. Retourne (événements triés, erreurs [(agenda, message)], nombre d'agendas).
    Un agenda en erreur n'empêche pas les autres."""
    cals = list_calendars(client_id, client_secret, exclude)
    tmin = _local_midnight(first).isoformat()
    tmax = _local_midnight(last + datetime.timedelta(days=1)).isoformat()

    def one(cal):
        items, page = [], None
        while True:
            params = {"timeMin": tmin, "timeMax": tmax, "singleEvents": "true", "orderBy": "startTime", "maxResults": 250,
                      "fields": "nextPageToken,items(id,summary,start,end,location,status,eventType,attendees(self,responseStatus))"}
            if page: params["pageToken"] = page
            data = _api_get(f"/calendars/{urllib.parse.quote(cal['id'], safe='')}/events", params, client_id, client_secret)
            items += [ev for ev in (_parse_event(i, cal["name"], cal.get("mine", True)) for i in data.get("items") or []) if ev]
            page = data.get("nextPageToken")
            if not page or len(items) >= 500: return items

    events, errors = [], []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(cals)))) as ex:
        futures = [(c, ex.submit(one, c)) for c in cals]
        for cal, fut in futures:
            try: events += fut.result()
            except AgendaError as err: errors.append((cal["name"], str(err)))
    events.sort(key=lambda ev: (_event_sort_key(ev)))
    return events, errors, len(cals)


def _event_sort_key(ev):
    s = ev["start"]
    return (s.isoformat() if ev["all_day"] else s.date().isoformat(), 0 if ev["all_day"] else 1, "" if ev["all_day"] else s.strftime("%H%M"), ev["title"].lower())


# =====================================================================================
#  Comprendre la demande (« j'ai quoi de prévu cette semaine ? »)
# =====================================================================================
_AGENDA_WORD = re.compile(r"\b(agenda|agendas|planning|calendrier)\b")
_APPT_WORD = re.compile(r"\b(rendez vous|rdv|rdvs|reunions?|evenements?|programme)\b")
_PREVU_WORD = re.compile(r"\bprevu\w*\b")
# « prévu » n'est une demande que dans une vraie question : « j'ai quoi de prévu », « qu'est-ce que j'ai de prévu », « ai-je quelque chose de prévu »
# (et pas « j'ai prévu de te dire merci »).
_PREVU_QUESTION = re.compile(r"\b(quoi|qu est ce|ai je|y a t il|y en a|combien|quels?|quelles?|j ai de prevu|de prevu)\b")
_ASK_WORD = re.compile(r"\b(quoi|quel|quels|quelle|quelles|combien|ai je|j ai|y a t il|y en a|qu est ce|ce que|consulte|montre|donne|regarde|lis|vois|affiche)\b")
_WRITE_WORD = re.compile(r"\b(ajoute|ajouter|cree|creer|supprime|supprimer|modifie|modifier|annule|annuler|deplace|deplacer|efface|envoie|ecris|ecrire|rappelle)\b")
_PERIOD_WORD = re.compile(r"\b(aujourd hui|ce matin|cet apres midi|ce soir|demain|apres demain|semaine|week end|weekend|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b")
_WEEKDAYS = {n: i for i, n in enumerate(JOURS)}


def parse_agenda_request(text, today=None):
    """Retourne {first, last, label, kind} si le message demande l'agenda, sinon None. kind : 'day' ou 'range'. Par défaut : aujourd'hui."""
    t = _norm(text)
    if not t or len(t.split()) > 25 or _WRITE_WORD.search(t): return None
    asks = bool(_ASK_WORD.search(t))
    wants = bool(_AGENDA_WORD.search(t)) or (bool(_APPT_WORD.search(t)) and asks and bool(_PERIOD_WORD.search(t) or _PREVU_WORD.search(t))) \
        or (bool(_PREVU_WORD.search(t)) and bool(_PREVU_QUESTION.search(t)))
    if not wants: return None
    today = today or datetime.date.today()
    wd = today.weekday()
    monday = today - datetime.timedelta(days=wd)
    if re.search(r"\bapres demain\b", t):
        d = today + datetime.timedelta(days=2); return {"first": d, "last": d, "label": "après-demain", "kind": "day"}
    if re.search(r"\bdemain\b", t):
        d = today + datetime.timedelta(days=1); return {"first": d, "last": d, "label": "demain", "kind": "day"}
    if re.search(r"\bsemaine prochaine\b|\bsemaine suivante\b|\bprochaine semaine\b", t):
        return {"first": monday + datetime.timedelta(days=7), "last": monday + datetime.timedelta(days=13), "label": "la semaine prochaine", "kind": "range"}
    if re.search(r"\bsemaine\b", t):
        return {"first": today, "last": monday + datetime.timedelta(days=6), "label": "cette semaine", "kind": "range"}
    if re.search(r"\bweek end\b|\bweekend\b", t):
        sat = today + datetime.timedelta(days=(5 - wd) % 7) if wd < 5 else today - datetime.timedelta(days=wd - 5)
        return {"first": sat, "last": sat + datetime.timedelta(days=1), "label": "ce week-end", "kind": "range"}
    for name, idx in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t) and not re.search(r"\baujourd hui\b", t):
            ahead = (idx - wd) % 7
            if ahead == 0 and re.search(r"\bprochain\b", t): ahead = 7
            d = today + datetime.timedelta(days=ahead)
            return {"first": d, "last": d, "label": f"{name} {d.day} {MOIS[d.month - 1]}", "kind": "day"}
    return {"first": today, "last": today, "label": "aujourd'hui", "kind": "day"}


# =====================================================================================
#  Mise en forme (texte affiché, phrase dite)
# =====================================================================================
def _day_title(d, today):
    full = f"{JOURS[d.weekday()]} {d.day} {MOIS[d.month - 1]}"
    if d == today: return f"Aujourd'hui · {full}"
    if d == today + datetime.timedelta(days=1): return f"Demain · {full}"
    return full.capitalize()


def _days_of(ev, first, last):
    """Jours (entre first et last) où apparaît l'événement : un événement « toute la journée » sur plusieurs jours apparaît chaque jour,
    un événement avec heures apparaît le jour où il commence (ou le 1er jour affiché s'il avait commencé avant)."""
    if ev["all_day"]:
        d, end = max(ev["start"], first), min(ev["end"] - datetime.timedelta(days=1), last)
        out = []
        while d <= end:
            out.append(d); d += datetime.timedelta(days=1)
        return out
    day = ev["start"].date()
    return [min(max(day, first), last)] if first <= day <= last or day < first else []


def _line(ev, day, show_calendar):
    """Un événement sur deux lignes : « • heure — titre », puis (en retrait) le lieu et le nom de l'agenda."""
    if ev["all_day"]:
        last_day = ev["end"] - datetime.timedelta(days=1)
        when = "Toute la journée"
        title = ev["title"] + (f" (jusqu'au {last_day.day:02d}/{last_day.month:02d})" if last_day > day else "")
    else:
        s, e = ev["start"], ev["end"]
        if s.date() < day: when = "En cours"
        elif e.date() > s.date(): when = f"{s:%H:%M} → {e:%H:%M} (le lendemain)"
        elif e > s: when = f"{s:%H:%M}–{e:%H:%M}"
        else: when = f"{s:%H:%M}"
        title = ev["title"]
    detail = [x for x in (ev["location"], ev["calendar"] if show_calendar else "") if x]
    head = f"•  {when}  —  {title}"
    return head + (("\n      " + "  ·  ".join(detail)) if detail else "")


def _when_title(ev, day):
    """(heure, titre) d'un événement pour un jour donné, comme dans _line."""
    head = _line(ev, day, False).split("\n")[0]                    # « •  09:00–10:30  —  Titre »
    when, _, title = head[3:].partition("  —  ")
    return when, title


def agenda_data(events, period, errors=(), calendars=1, today=None):
    """Agenda sous forme de données (pour l'interface : une capsule par jour) :
    {"days": [{"title", "events": [{"when", "title", "location", "calendar", "mine"}]}], "empty": phrase ou None, "more": phrase ou None,
     "count": phrase ou None, "errors": [phrases], "show_calendar": bool, "mixed": bool (des agendas à toi ET partagés)}."""
    today = today or datetime.date.today()
    first, last, label = period["first"], period["last"], period["label"]
    by_day, shown = {}, 0
    for ev in events:
        for d in _days_of(ev, first, last):
            by_day.setdefault(d, []).append(ev)
    count = sum(len(v) for v in by_day.values())
    days, more = [], None
    for d in sorted(by_day):
        items = []
        for ev in sorted(by_day[d], key=_event_sort_key):
            if shown >= MAX_EVENTS: break
            when, title = _when_title(ev, d)
            items.append({"when": when, "title": title, "location": ev["location"], "calendar": ev["calendar"], "mine": ev.get("mine", True)})
            shown += 1
        if items: days.append({"title": _day_title(d, today), "events": items})
    if count > shown:
        rest = count - shown
        more = f"… et {rest} autre{'s' if rest > 1 else ''} événement{'s' if rest > 1 else ''}."
    kinds = {ev.get("mine", True) for ev in events}
    return {"days": days, "empty": None if count else f"Rien de prévu {label}.", "more": more,
            "count": f"{count} événement{'s' if count > 1 else ''} {label}." if count else None,
            "errors": [f"⚠ Agenda illisible — {name} : {msg}" for name, msg in errors], "show_calendar": calendars > 1, "mixed": len(kinds) > 1}


def agenda_text(events, period, errors=(), calendars=1, today=None):
    """Texte (copie, secours) : événements regroupés par jour (« toute la journée » d'abord, puis par heure). Rien de prévu → phrase simple."""
    data = agenda_data(events, period, errors, calendars, today)
    lines = []
    if data["empty"]: lines.append(data["empty"])
    for day in data["days"]:
        lines.append(day["title"])
        lines.append("─" * min(len(day["title"]) + 4, 34))
        for ev in day["events"]:
            lines.append(f"•  {ev['when']}  —  {ev['title']}")
            detail = [x for x in (ev["location"], ev["calendar"] if data["show_calendar"] else "") if x]
            if detail: lines.append("      " + "  ·  ".join(detail))
        lines.append("")
    if data["more"]: lines.append(data["more"])
    if data["count"]: lines.append(data["count"])
    lines += data["errors"]
    return "\n".join(lines).strip()


def _say_time(dt):
    h, m = dt.hour, dt.minute
    if h == 0 and m == 0: return "minuit"
    if h == 12 and m == 0: return "midi"
    return f"{h} heures" + (f" {m}" if m else "")


def _short_title(title, limit=45):
    title = " ".join(title.split())
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0]


def agenda_spoken(events, period, errors=(), today=None, now=None):
    """Phrase courte dite à voix haute. Un jour : les événements dans l'ordre (4 au plus). Plusieurs jours : le nombre et les 3 prochains.
    Le détail est à l'écran."""
    today = today or datetime.date.today()
    now = now or datetime.datetime.now().astimezone()
    first, last, label = period["first"], period["last"], period["label"]
    entries = []                                              # (jour, événement)
    for ev in events:
        for d in _days_of(ev, first, last): entries.append((d, ev))
    entries.sort(key=lambda x: (x[0],) + _event_sort_key(x[1])[1:])
    suffix = " Une partie de l'agenda n'a pas pu être lue." if errors else ""
    if not entries:
        return (f"Tu n'as rien de prévu {label}." if not errors else "Je n'ai pas pu lire tout ton agenda, regarde l'écran.")
    def describe(ev, day, with_day):
        when = "toute la journée" if ev["all_day"] else f"à {_say_time(ev['start'])}"
        where = ""
        if with_day: where = " aujourd'hui" if day == today else " demain" if day == today + datetime.timedelta(days=1) else f" {JOURS[day.weekday()]}"
        return f"{_short_title(ev['title'])}{where} {when}"
    n = len(entries)
    plural = "s" if n > 1 else ""
    def enumerate_(listed):
        return ", ".join(listed[:-1]) + (" et " if len(listed) > 1 else "") + listed[-1]
    if period["kind"] == "day":
        listed = [describe(ev, d, False) for d, ev in entries[:4]]
        body = enumerate_(listed) if n <= 4 else ", ".join(listed) + f", et {n - 4} autre{'s' if n - 4 > 1 else ''}"
        return f"{label.capitalize()} tu as {n} événement{plural} : {body}." + suffix
    def still_to_come(d, ev):
        if d > today: return True
        return (ev["all_day"] and d == today) or (not ev["all_day"] and ev["end"] >= now)
    upcoming = [(d, ev) for d, ev in entries if still_to_come(d, ev)] or entries
    listed = [describe(ev, d, True) for d, ev in upcoming[:3]]
    return f"{label.capitalize()} tu as {n} événement{plural}. Les prochains : {enumerate_(listed)}. Le détail est à l'écran." + suffix
