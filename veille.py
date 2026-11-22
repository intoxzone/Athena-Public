"""« News IA » d'Athena (module veille.py) : repère les nouveaux modèles qui pourraient convenir à ta machine, et quelques articles en français.

Lecture seule, adresses publiques seulement (aucune donnée personnelle n'est envoyée), AUCUN modèle de langage : les cartes sont fabriquées
par le code à partir des métadonnées (jamais d'invention possible, jamais d'instruction cachée dans une page).

Sources :
  1. la bibliothèque d'Ollama (nouveautés, mentions vision/tools/thinking, taille réelle de chaque version via le registre) ;
  2. Hugging Face (API publique) : nouveaux modèles des éditeurs de la liste `editeurs` ;
  3. des flux RSS d'articles en français (`sources_articles`), filtrés par mots-clés ;
  + une surveillance spéciale : un petit modèle Qwen d'une génération récente (réglages qwen_min / qwen_max_params).

Les réglages viennent de config.toml (section [veille]) : engine.py recopie les valeurs ci-dessous au démarrage.
"""
import re, os, json, time, datetime, threading, unicodedata, html as _html
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN_PATH = os.path.join(HERE, "veille_vue.json")          # ce que tu as déjà vu (personnel, ignoré par Git)

# ----------------------------------------------------------------------------- réglages (valeurs par défaut, voir config.toml)
ACTIVE = True
CHECK_AT_START = True                 # vérification discrète au démarrage d'Athena (met à jour le compteur de la pastille)
DAYS = 30                             # on regarde les modèles des N derniers jours
ARTICLE_DAYS = 7                      # ... et les articles des N derniers jours
SIZE_OK_GB = 6.0                      # taille du fichier Q4 : ✅ jusqu'à ici
SIZE_LIMIT_GB = 6.5                   # ⚠️ jusqu'à ici, ❌ au-delà (ton Qwen 5,24 Go occupe ~6,4 Go de VRAM)
MAX_MODELS = 8
MAX_ARTICLES = 6
TIMEOUT = 10.0                        # secondes max par requête
HIDE_TOO_BIG = True                   # n'affiche pas les modèles trop gros (sauf un petit Qwen récent)
PUBLISHERS = ["Qwen", "mistralai", "google", "meta-llama", "microsoft", "ibm-granite", "nvidia", "HuggingFaceTB", "LiquidAI", "allenai",
              "deepseek-ai", "openai"]
ARTICLE_SOURCES = ["ActuIA=https://www.actuia.com/feed/", "Next=https://next.ink/feed/", "Korben=https://korben.info/feed",
                   "Numerama=https://www.numerama.com/feed/"]
QWEN_MIN = 3.6                        # surveillance spéciale : génération Qwen minimale...
QWEN_MAX_PARAMS = 10.0                # ... et taille maximale (milliards de paramètres) d'un « petit Qwen récent »

CACHE_MINUTES = 30
MAX_BYTES = 3_000_000
UA = "Athena-veille (assistant local)"
Q4_GB_PER_BILLION = 0.58              # estimation de la taille d'un fichier Q4_K_M (5,24 Go pour 9 milliards)

SPOKEN_NEW = "Je t'affiche les dernières news IA à l'écran."
SPOKEN_NONE = "Rien de nouveau depuis tes dernières news IA. Je t'affiche les derniers éléments connus à l'écran."
SPOKEN_FAIL = "Je n'ai pas pu consulter les sources des news IA."
NAME = "News IA"                                                 # nom affiché (pastille, titres)


class VeilleError(Exception):
    """Une source n'a pas pu être lue (message en français, sans adresse personnelle)."""


# ----------------------------------------------------------------------------- reconnaissance de la demande
def _norm(text):
    t = unicodedata.normalize("NFKD", str(text).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


_REQ = [re.compile(p) for p in (
    r"\bnews (de l |des |sur l |sur les )?(ia|ai|llm|modeles?)\b",          # « news IA », « les news de l'IA », « mes news modèles »
    r"\b(ia|ai|llm) news\b",
    # anciens mots (« veille »), gardés comme alias
    r"\b(fais|fait|lance|lancer|donne|montre|affiche|ouvre|rafraichis)\b.{0,25}\bveille\b",
    r"\bma veille\b",
    r"\bveille (ia|modeles?|techno\w*|llm)\b",
)]
_REQ_NEWS = re.compile(r"\b(nouveaux?|nouveautes?|nouvelles?|quoi de neuf|du neuf|derniers?|dernieres?)\b.{0,35}\b(modeles?|llm)\b|"
                       r"\by a t il\b.{0,25}\bmodeles?\b")
_AI_CTX = re.compile(r"\b(ia|llm|ollama|qwen|mistral|gemma|llama|langage|config|configuration|intelligence artificielle|machine|pc)\b")


def is_veille_request(text):
    """« donne-moi les news IA », « mes news IA », « quoi de neuf côté modèles d'IA ? », « y a-t-il de nouveaux modèles pour ma config ? »
    (et les anciens « fais ma veille », « veille IA »)."""
    t = _norm(text)
    if not ACTIVE or not t or len(t.split()) > 16: return False
    if re.search(r"\ben veille\b|\bmets? toi\b|\bmet toi\b", t): return False       # « Mets-toi en veille » : commande vocale existante
    if any(p.search(t) for p in _REQ): return True
    return bool(_REQ_NEWS.search(t) and _AI_CTX.search(t))


# ----------------------------------------------------------------------------- accès réseau (un seul point : facile à remplacer dans les tests)
def _get(url, headers=None):
    """Télécharge une adresse http(s) publique (taille limitée). Retourne des octets. Lève VeilleError."""
    if not str(url).lower().startswith(("http://", "https://")):
        raise VeilleError("adresse non valide")
    try:
        r = requests.get(url, headers={"User-Agent": UA, **(headers or {})}, timeout=TIMEOUT, stream=True)
    except requests.exceptions.RequestException:
        raise VeilleError("site injoignable")
    try:
        if r.status_code != 200: raise VeilleError(f"réponse {r.status_code}")
        data = b""
        try:
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) > MAX_BYTES: raise VeilleError("réponse trop volumineuse")
        except requests.exceptions.RequestException:
            raise VeilleError("connexion interrompue")
        return data
    finally:
        r.close()


def _get_json(url, headers=None):
    try: return json.loads(_get(url, headers).decode("utf-8", "replace"))
    except ValueError: raise VeilleError("réponse illisible")


# ----------------------------------------------------------------------------- aides
_MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}


def _today(): return datetime.date.today()


def _short(text, n=160):
    t = " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", str(text))).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def parse_params(name):
    """Taille d'un modèle en milliards de paramètres d'après son nom (« Qwen3.5-9B » → 9, « -2.4T- » → 2400), sinon None.
    Les nombres actifs d'un modèle « mixture of experts » (« A3B ») et les versions (« 3.8 ») ne comptent pas."""
    best = None
    for m in re.finditer(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([BTbt])(?![A-Za-z0-9])", str(name)):
        v = float(m.group(1)) * (1000 if m.group(2) in "Tt" else 1)
        best = v if best is None else max(best, v)
    return best


def qwen_generation(name):
    m = re.search(r"qwen[ _-]?(\d+(?:\.\d+)?)", str(name).lower())
    return float(m.group(1)) if m else None


def fit_status(size_gb):
    if size_gb is None: return "unknown"
    return "ok" if size_gb <= SIZE_OK_GB else "limit" if size_gb <= SIZE_LIMIT_GB else "no"


def _is_star(name, params):
    g = qwen_generation(name)
    return bool(g is not None and g >= QWEN_MIN and params is not None and params <= QWEN_MAX_PARAMS)


# ----------------------------------------------------------------------------- source 1 : bibliothèque d'Ollama
def parse_ollama_list(page):
    """Liste des modèles de la page « nouveautés » d'Ollama : [{name, date, caps, desc, downloads}]."""
    out = []
    for li in re.findall(r"<li\b.*?</li>", page, re.S):
        m = re.search(r'href="/library/([A-Za-z0-9._-]+)"', li)
        if not m: continue
        d = re.search(r'title="Updated ([A-Za-z]+) (\d{1,2}), (\d{4})"', li)
        date = None
        if d and d.group(1) in _MONTHS:
            try: date = datetime.date(int(d.group(3)), _MONTHS[d.group(1)], int(d.group(2)))
            except ValueError: date = None
        desc = re.search(r'<p class="[^"]*">(.*?)</p>', li, re.S)
        dl = re.search(r'title="([\d,]+) downloads"', li)
        out.append({"name": m.group(1), "date": date, "caps": re.findall(r">(vision|tools|thinking|embedding|audio|cloud)</span>", li),
                    "desc": _short(desc.group(1)) if desc else "", "downloads": int(dl.group(1).replace(",", "")) if dl else None})
    return out


def ollama_tags(name):
    """Versions « taille » d'un modèle (8b, 0.8b, 27b…), triées par taille ; ['latest'] s'il n'y en a pas."""
    page = _get(f"https://ollama.com/library/{name}").decode("utf-8", "replace")
    tags = sorted(set(re.findall(rf'/library/{re.escape(name)}:([A-Za-z0-9._-]+)', page)))
    sized = [t for t in tags if re.fullmatch(r"\d+(?:\.\d+)?[bm]", t)]
    sized.sort(key=lambda t: float(t[:-1]) * (0.001 if t[-1] == "m" else 1))
    return sized or ["latest"]


def ollama_size_gb(name, tag):
    """Taille réelle du fichier modèle (Go), lue dans le registre d'Ollama ; None si introuvable."""
    try:
        m = _get_json(f"https://registry.ollama.ai/v2/library/{name}/manifests/{tag}",
                      {"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        size = sum(int(l.get("size") or 0) for l in m.get("layers") or [] if re.search(r"image\.(model|tensor)", str(l.get("mediaType"))))
        return round(size / 1024 ** 3, 2) if size else None
    except (VeilleError, AttributeError, TypeError, ValueError):
        return None


def _ollama_item(entry):
    name = entry["name"]
    tags = ollama_tags(name)
    sizes = [(t, ollama_size_gb(name, t)) for t in tags[:8]]
    sizes = [(t, s) for t, s in sizes if s is not None]
    star = False
    if qwen_generation(name) is not None and qwen_generation(name) >= QWEN_MIN:
        star = any(re.fullmatch(r"\d+(?:\.\d+)?b", t) and float(t[:-1]) <= QWEN_MAX_PARAMS for t in tags)
    fits = [(t, s) for t, s in sizes if s <= SIZE_LIMIT_GB]
    if fits: tag, size = max(fits, key=lambda x: x[1])              # la plus grande version qui tient
    elif sizes: tag, size = min(sizes, key=lambda x: x[1])
    else: tag, size = (tags[0] if tags else "latest"), None
    if size is None and "cloud" in entry["caps"]: return None          # modèle seulement disponible dans le cloud d'Ollama : inutile pour ta machine
    return {"key": f"ollama:{name}", "name": f"{name}:{tag}", "by": "Ollama", "source": "Ollama", "date": entry["date"].isoformat() if entry["date"] else "",
            "size_gb": size, "estimated": False, "fit": fit_status(size), "caps": [c for c in entry["caps"] if c in ("vision", "tools", "thinking", "audio")],
            "desc": entry["desc"], "downloads": entry["downloads"], "url": f"https://ollama.com/library/{name}", "pull": f"ollama pull {name}:{tag}",
            "star": star}


def ollama_models():
    entries, seen = [], set()
    for url in ("https://ollama.com/search?o=newest", "https://ollama.com/search?o=newest&page=2"):
        try: page = _get(url).decode("utf-8", "replace")
        except VeilleError:
            if not entries: raise
            break
        for e in parse_ollama_list(page):
            if e["name"] not in seen: seen.add(e["name"]); entries.append(e)
    cutoff = _today() - datetime.timedelta(days=DAYS)
    keep = [e for e in entries if (e["date"] is None or e["date"] >= cutoff) and "embedding" not in e["caps"] and "embed" not in e["name"].lower()]
    keep = keep[:24]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda e: _safe(lambda: _ollama_item(e)), keep))
    return [r for r in results if r]


def _safe(fn):
    try: return fn()
    except (VeilleError, ValueError, KeyError, TypeError): return None


# ----------------------------------------------------------------------------- source 2 : Hugging Face
_HF_SKIP = re.compile(r"FP8|FP4|NVFP4|AWQ|GPTQ|INT[48]|bnb|MLX|ONNX|WebGPU|nfe\d|spec\b|Base\b|Pretrain|Image|Embedding|Rerank|Guard|TTS|ASR|SAE|Whisper|Diffusion|Video|OCR", re.I)
MAX_PER_PUBLISHER = 2                 # Hugging Face : au plus N modèles par éditeur (évite les séries de variantes quasi identiques)


def _hf_author(author):
    items = _get_json(f"https://huggingface.co/api/models?author={author}&sort=createdAt&direction=-1&limit=40")
    cutoff = _today() - datetime.timedelta(days=DAYS)
    out = []
    for m in items if isinstance(items, list) else []:
        try:
            rid, created = str(m["id"]), str(m.get("createdAt", ""))[:10]
            if not created or datetime.date.fromisoformat(created) < cutoff: continue
            if m.get("pipeline_tag") not in ("text-generation", "image-text-to-text") or _HF_SKIP.search(rid.split("/")[-1]): continue
        except (KeyError, ValueError, TypeError):
            continue
        params = parse_params(rid.split("/")[-1])
        size = round(params * Q4_GB_PER_BILLION, 2) if params is not None and params < 1000 else None
        out.append({"key": f"hf:{rid}", "name": rid.split("/")[-1], "by": rid.split("/")[0], "source": "Hugging Face", "date": created, "size_gb": size,
                    "estimated": True, "fit": fit_status(size), "caps": ["vision"] if m.get("pipeline_tag") == "image-text-to-text" else [],
                    "desc": "", "downloads": m.get("downloads"), "url": f"https://huggingface.co/{rid}", "pull": "",
                    "star": _is_star(rid.split("/")[-1], params)})
    out.sort(key=lambda x: (not x["star"], -(x["downloads"] or 0)))
    return [x for x in out if x["star"]] + [x for x in out if not x["star"]][:MAX_PER_PUBLISHER]


def hf_models():
    errors, out = 0, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(lambda a: _safe(lambda: _hf_author(a)), PUBLISHERS):
            if res is None: errors += 1
            else: out += res
    if errors == len(PUBLISHERS): raise VeilleError("Hugging Face injoignable")
    return out


# ----------------------------------------------------------------------------- source 3 : articles en français (flux RSS)
_ARTICLE_KW = re.compile(r"\b(llm|ollama|qwen|mistral|llama|gemma|deepseek|granite|nemotron|gguf|hugging ?face|ia locale|ia open|poids ouverts|open[- ]weights?|"
                         r"mod[èe]les? (de langage|d['’]ia|ouverts?|open|locaux|locale))\b", re.I)


def parse_feed(raw):
    """Articles d'un flux RSS/Atom : [{title, link, date(datetime|None), text}]. Refuse les flux contenant des entités XML (protection)."""
    if re.search(rb"<!DOCTYPE|<!ENTITY", raw[:4000] + raw, re.I): raise VeilleError("flux refusé (entités XML)")
    try: root = ET.fromstring(raw)
    except ET.ParseError: raise VeilleError("flux illisible")
    out = []
    for it in root.iter():
        if it.tag.split("}")[-1] not in ("item", "entry"): continue
        d = {}
        for c in it:
            tag = c.tag.split("}")[-1]
            if tag == "link" and c.attrib.get("href"): d.setdefault("link", c.attrib["href"])
            elif tag in ("title", "link", "pubDate", "updated", "published", "description", "summary") and tag not in d: d[tag] = (c.text or "").strip()
        link = d.get("link", "")
        if not link.lower().startswith(("http://", "https://")): continue
        when = None
        raw_date = d.get("pubDate") or d.get("published") or d.get("updated") or ""
        for parse in (parsedate_to_datetime, lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))):
            try: when = parse(raw_date); break
            except (TypeError, ValueError): continue
        out.append({"title": _short(d.get("title", ""), 140), "link": link, "date": when, "text": _short(d.get("description") or d.get("summary") or "", 300)})
    return out


def articles():
    cutoff = _today() - datetime.timedelta(days=ARTICLE_DAYS)
    out, errors = [], []
    for spec in ARTICLE_SOURCES:
        name, _, url = spec.partition("=")
        if not url: continue
        try:
            for a in parse_feed(_get(url.strip())):
                day = a["date"].date() if a["date"] else None
                if (day is None or day >= cutoff) and _ARTICLE_KW.search(f"{a['title']} {a['text']}"):
                    out.append({"key": f"art:{a['link']}", "title": a["title"], "source": name.strip(), "date": day.isoformat() if day else "", "url": a["link"]})
        except VeilleError as e:
            errors.append(f"{name.strip()} : {e}")
    return out, errors


# ----------------------------------------------------------------------------- mémoire de ce qui a été vu
def load_seen():
    try:
        with open(SEEN_PATH, encoding="utf-8") as f: d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_seen(seen):
    try:
        tmp = SEEN_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(seen, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SEEN_PATH)
    except OSError:
        pass


# ----------------------------------------------------------------------------- collecte, tri, affichage
_cache = {"time": 0.0, "raw": None}
_lock = threading.Lock()


def collect():
    """Interroge toutes les sources (une source en échec n'empêche pas les autres). Retourne les données brutes."""
    errors, models, arts = [], [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_ollama, f_hf, f_art = ex.submit(ollama_models), ex.submit(hf_models), ex.submit(articles)
        for label, fut in (("Ollama", f_ollama), ("Hugging Face", f_hf)):
            try: models += fut.result()
            except VeilleError as e: errors.append(f"{label} : {e}")
        try:
            arts, art_errors = f_art.result()
            errors += art_errors
        except VeilleError as e:
            errors.append(f"Articles : {e}")
    return {"models": models, "articles": arts, "errors": errors, "checked": datetime.datetime.now().isoformat(timespec="seconds")}


_FIT_ORDER = {"ok": 0, "limit": 1, "unknown": 2, "no": 3}


def annotate(raw):
    """Données prêtes à afficher : marque ce qui est nouveau (jamais vu), trie, limite. Ne modifie pas `raw`."""
    seen = load_seen()
    seen_models, seen_articles = seen.get("models", {}), seen.get("articles", {})
    models = [dict(m, new=m["key"] not in seen_models) for m in raw["models"]]
    hidden = 0
    if HIDE_TOO_BIG:
        hidden = sum(1 for m in models if m["fit"] == "no" and not m["star"])
        models = [m for m in models if m["fit"] != "no" or m["star"]]
    models.sort(key=lambda m: (not m["star"], not m["new"], _FIT_ORDER.get(m["fit"], 2), "tools" not in m["caps"], m["source"] != "Ollama", -int(m["date"].replace("-", "") or 0)))
    arts = [dict(a, new=a["key"] not in seen_articles) for a in raw["articles"]]
    arts.sort(key=lambda a: (not a["new"], -int(a["date"].replace("-", "") or 0)))
    new_models, new_arts = [m for m in models if m["new"]], [a for a in arts if a["new"]]
    shown_models, shown_arts = new_models[:MAX_MODELS], new_arts[:MAX_ARTICLES]
    recap = not shown_models and not shown_arts                    # rien de nouveau : on montre quand même les derniers éléments connus
    if recap: shown_models, shown_arts = models[:4], arts[:3]
    return {"models": shown_models, "articles": shown_arts, "errors": list(raw["errors"]), "checked": raw["checked"], "hidden_big": hidden,
            "new_count": 0 if recap else len(shown_models) + len(shown_arts), "recap": recap}


def get_data(force=False):
    """Données de la veille, depuis le cache (30 min) ou en interrogeant les sources. Lève VeilleError si TOUTES les sources échouent."""
    with _lock:
        if force or _cache["raw"] is None or time.time() - _cache["time"] > CACHE_MINUTES * 60:
            raw = collect()
            if not raw["models"] and not raw["articles"] and raw["errors"]:
                raise VeilleError("aucune source n'a répondu : " + " ; ".join(raw["errors"][:3]))
            _cache.update(time=time.time(), raw=raw)
        raw = _cache["raw"]
    return annotate(raw)


def pending_count():
    """Nombre de nouveautés (compteur de la pastille), d'après le cache ; 0 si la vérification n'a pas encore été faite."""
    return annotate(_cache["raw"])["new_count"] if _cache["raw"] else 0


def mark_seen(data):
    """Enregistre ce qui vient d'être affiché : ce ne sera plus « nouveau »."""
    seen = load_seen()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    seen.setdefault("models", {}).update({m["key"]: now for m in data["models"]})
    seen.setdefault("articles", {}).update({a["key"]: now for a in data["articles"]})
    seen["last"] = now
    save_seen(seen)


# ----------------------------------------------------------------------------- texte (copie / secours)
_FIT_TEXT = {"ok": "✅ tient", "limit": "⚠️ limite", "no": "❌ trop gros", "unknown": "taille inconnue"}


def _size_text(m):
    if m["size_gb"] is None: return "taille inconnue"
    return f"{'≈ ' if m['estimated'] else ''}{m['size_gb']:.1f} Go"


def veille_text(data):
    lines = []
    if data["recap"]: lines += ["Rien de nouveau depuis tes dernières news IA. Derniers éléments connus :", ""]
    if data["models"]:
        lines += ["Modèles", "───────"]
        for m in data["models"]:
            lines.append(f"{'⭐ ' if m['star'] else ''}{'[NOUVEAU] ' if m['new'] else ''}{m['name']} — {m['by']} — {_size_text(m)} — {_FIT_TEXT[m['fit']]}"
                         + (f" — {', '.join(m['caps'])}" if m["caps"] else "") + f" — {m['date']}")
            if m["desc"]: lines.append("   " + m["desc"])
            if m["pull"]: lines.append("   " + m["pull"])
            lines.append("   " + m["url"])
        lines.append("")
    if data["articles"]:
        lines += ["Articles", "────────"]
        for a in data["articles"]:
            lines += [f"{'[NOUVEAU] ' if a['new'] else ''}{a['title']} — {a['source']} — {a['date']}", "   " + a["url"]]
        lines.append("")
    if data["hidden_big"]: lines.append(f"{data['hidden_big']} modèle(s) trop gros pour ta carte (plus de {SIZE_LIMIT_GB:g} Go) non affichés.")
    lines += [f"⚠ Source indisponible — {e}" for e in data["errors"]]
    return "\n".join(lines).strip() or "Rien à afficher pour le moment."


def spoken(data):
    return SPOKEN_NONE if data["recap"] else SPOKEN_NEW
