# Athena

Assistant vocal **local** pour Windows, en français. Athena écoute (ou lit ce que tu tapes), réfléchit avec un modèle de langage qui tourne **sur ton PC** via [Ollama](https://ollama.com), puis répond à l'écran et à voix haute.

Le but : un assistant utile au quotidien qui garde tes données chez toi. Le modèle, la reconnaissance vocale et la synthèse vocale sont tous locaux ; seules les recherches web, la météo et les services que tu actives (mails, agenda) sortent de ta machine.

> Projet personnel, en développement actif. Il est utilisé tous les jours par son auteur, mais n'a été testé que sur une seule configuration (voir « Matériel »). Attends-toi à devoir adapter quelques réglages.

## Ce qu'Athena sait faire

- **Deux modes** : écrit (clavier, par défaut) et oral (micro). Le micro-interrupteur bascule de l'un à l'autre ; la voix n'est chargée qu'à la première activation du micro.
- **Réveil par la voix** : dis « Athena », puis ta question. Commandes « stop », « mets-toi en veille », « déconnecte ».
- **Recherche web** avec sources cliquables sous la réponse (via ta propre instance [SearXNG](https://docs.searxng.org)).
- **Lecture d'une page web** : colle un lien ou clique sur « Lire » sous une source, Athena résume la page.
- **Mails (lecture seule)** : « résume mes mails ». Le résumé n'est jamais lu à voix haute, seul le nombre de mails non lus l'est.
- **Agenda Google (lecture seule)** : « j'ai quoi de prévu demain ? ».
- **Offres d'emploi** : « cherche des offres de cariste » (recherche web, ou API France Travail si tu y as accès).
- **Routine du matin** : météo ([Open-Meteo](https://open-meteo.com)), agenda, mails, offres d'emploi et actualités, enchaînés en un seul message.
- **News IA** : repère les nouveaux modèles de langage qui tiennent sur ta carte graphique et des articles en français.
- **Historique des discussions** dans une barre latérale (3 jours de conservation, discussions épinglables).

Les outils (mails, agenda, emploi, pages web, routine) sont déclenchés **par le code**, pas par le modèle : un petit modèle local appelle mal les outils, et ça évite qu'un texte piégé dans un mail ou une page lui donne des ordres.

## Ce qui tourne en local

| Rôle | Composant |
|---|---|
| Modèle de langage | [Ollama](https://ollama.com) (`qwen3.5:9b` par défaut) |
| Transcription de la voix | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-server`, sur le processeur) |
| Mot de réveil | [Vosk](https://alphacephei.com/vosk/) (modèle français léger) |
| Fin de phrase | [Silero VAD](https://github.com/snakers4/silero-vad) |
| Synthèse vocale | [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx) (voix française `ff_siwis`) |
| Interface | [Flet](https://flet.dev) |

## Matériel et système

- **Windows 10/11 uniquement** (registre Windows, lanceurs `.bat`, démarrage d'Ollama).
- **Python 3.12** (le fichier de configuration utilise `tomllib`).
- Testé sur : Ryzen 7 5700X, 32 Go de RAM, AMD Radeon RX 6650 XT **8 Go**. Le modèle de 9 milliards de paramètres occupe environ 6,4 Go de VRAM : sur une carte de 8 Go, c'est la ressource critique. Whisper tourne donc volontairement sur le processeur pour laisser toute la VRAM au modèle.
- Avec une autre carte, change `modele` et `taille_contexte` dans la section `[llm]` de la configuration.

## Installation

### 1. Organisation des dossiers

Athena cherche ses modèles dans le **dossier parent** du code. Crée une arborescence comme celle-ci (le nom du dossier racine est libre) :

```
Athena/
├─ app/                 ← ce dépôt
├─ venv/                ← environnement Python
├─ whisper/             ← ggml-small-q5_1.bin (modèle Whisper)
├─ whisper-cpu/         ← build processeur de whisper.cpp, avec whisper-server.exe
├─ kokoro/              ← kokoro-v1.0.onnx et voices-v1.0.bin
├─ vosk/                ← vosk-model-small-fr-0.22/
└─ silero/              ← silero_vad.onnx
```

### 2. Récupérer les modèles

- **Ollama** : installe-le depuis [ollama.com](https://ollama.com), puis `ollama pull qwen3.5:9b`.
- **Whisper** : binaires Windows (`whisper-bin-x64.zip`) dans les [versions de whisper.cpp](https://github.com/ggml-org/whisper.cpp/releases), à dézipper dans `whisper-cpu/` ; modèle `ggml-small-q5_1.bin` sur [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp), à mettre dans `whisper/`.
- **Kokoro** : `kokoro-v1.0.onnx` et `voices-v1.0.bin` dans les versions du dépôt [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx).
- **Vosk** : `vosk-model-small-fr-0.22` sur la [page des modèles Vosk](https://alphacephei.com/vosk/models).
- **Silero VAD** : `silero_vad.onnx` dans le dépôt [silero-vad](https://github.com/snakers4/silero-vad).

### 3. Python

Dans PowerShell, depuis le dossier racine :

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

### 4. Configuration

```powershell
cd app
copy config.exemple.toml config.toml
```

Ouvre `config.toml` : le fichier est commenté ligne par ligne. Tout réglage absent garde sa valeur par défaut, et une erreur de saisie n'empêche jamais Athena de démarrer (un message `[Config]` s'affiche).

À régler en priorité :
- ta recherche web : `[recherche] url` (adresse de ton SearXNG) et `utilisateur` ; tant qu'ils ne sont pas remplis, la recherche web est désactivée et Athena le dit ;
- ta ville et tes coordonnées : `[emploi]` et `[routine]` (Paris par défaut) ;
- ton prénom pour la salutation vocale : `[ecoute] prenom` (facultatif) ;
- le nom de ton micro dans les profils `[profils.N]`, pour ajuster le volume (facultatif).

Les chemins des modèles et d'Ollama ont des valeurs par défaut : ne les change que si tes dossiers sont ailleurs.

### 5. Lancer

- **Avec la console** (pour voir les messages) : double-clic sur `Athena-console.bat`.
- **Sans console** : lance `Athena.pyw` avec le `pythonw.exe` du venv ; les messages vont dans `athena.log`. Tu peux en faire un raccourci.

Si Ollama ne tourne pas, Athena lance son serveur toute seule (`ollama serve`, sans fenêtre). À la fermeture, elle décharge le modèle de la mémoire vidéo.

## Secrets

**Aucun mot de passe n'est écrit dans un fichier.** Ils se déclarent comme variables d'environnement Windows, une seule fois :

```powershell
setx ATHENA_SEARX_PASSWORD "ton-mot-de-passe"
```

puis ferme et rouvre ton terminal. Variables utilisées :

| Variable | Pour quoi |
|---|---|
| `ATHENA_SEARX_PASSWORD` (et `ATHENA_SEARX_USER`, ou `[recherche] utilisateur`) | recherche web (authentification basique de ton SearXNG) |
| `ATHENA_MAIL_1_PASSWORD`, `ATHENA_MAIL_2_PASSWORD`… | une variable par boîte mail déclarée dans `config.toml` |
| `ATHENA_GOOGLE_CLIENT_ID`, `ATHENA_GOOGLE_CLIENT_SECRET` | agenda Google (puis `Connecter-agenda.bat`, une seule fois) |
| `ATHENA_FT_ID`, `ATHENA_FT_SECRET` | API France Travail (facultatif) |

Gmail demande un « mot de passe d'application » ; Proton passe par Proton Mail Bridge. Les détails sont dans `config.exemple.toml`.

### Agenda Google (facultatif)

Athena lit tes agendas avec le seul droit `calendar.readonly`. Il faut ta propre application Google :

1. Dans la [console Google Cloud](https://console.cloud.google.com), crée un projet et active l'API **Google Calendar**.
2. Crée un identifiant OAuth de type **application de bureau** et note son identifiant client et son secret.
3. Définis `ATHENA_GOOGLE_CLIENT_ID` et `ATHENA_GOOGLE_CLIENT_SECRET` (commande `setx`, voir plus haut), puis rouvre ton terminal.
4. Lance `Connecter-agenda.bat` une seule fois : ton navigateur s'ouvre, tu acceptes l'accès en lecture.
5. Pour que l'autorisation n'expire pas au bout de 7 jours, passe l'application de « Test » à « En production » dans l'écran de consentement OAuth.

L'autorisation est gardée dans `agenda_token.json`, un fichier **secret** que Git ignore.

## Confidentialité

- Le modèle, la voix et la transcription tournent sur ton PC.
- **Mails et agenda** : accès en lecture seule (Athena ne peut ni envoyer, ni supprimer, ni marquer « lu »). Leur contenu n'est ni enregistré dans l'historique, ni écrit dans les logs. L'agenda est fabriqué par le code sans passer par le modèle.
- **Sorties vers Internet** : ton instance SearXNG, Open-Meteo (seules les coordonnées de la météo sont envoyées), le serveur IMAP de tes boîtes, Google Calendar si tu l'actives, et les listes publiques de modèles et d'articles pour « News IA ».
- Les fichiers personnels (`config.toml`, `history.json`, `prefs.json`, `agenda_token.json`, `athena.log`) sont ignorés par Git. Ne les publie jamais.

## Limites connues

- **La recherche web demande ta propre instance SearXNG** (format JSON activé) avec authentification basique : le code attend actuellement un mot de passe. Sans elle, Athena répond sans chercher.
- Windows seulement.
- Un modèle de 9 milliards de paramètres reste faillible : il peut se tromper ou inventer. Une consigne d'honnêteté et les sources affichées limitent le problème sans le supprimer.
- Le mot « stop » n'est pas entendu pendant qu'Athena parle dans des haut-parleurs (sa propre voix le couvre) ; il l'est entre deux phrases, et le bouton Stop marche toujours.
- L'API France Travail est désactivée par défaut (`source = "web"`) : elle a été écrite d'après la documentation, sans pouvoir être validée avec de vraies clés.
- Whisper `small` se trompe parfois sur les noms propres ; l'option `amorce` de la configuration aide à l'orienter.

## Structure du code

| Fichier | Rôle |
|---|---|
| `ui.py` | interface Flet |
| `engine.py` | moteur : écoute, transcription, modèle, voix, outils |
| `agenda.py`, `connecter_agenda.py` | Google Agenda (OAuth, lecture seule) |
| `veille.py` | « News IA » |
| `history.py` | historique des discussions |
| `logs.py` | horodatage de la console |
| `Athena.pyw` | lanceur sans console |

## Licence

[MIT](LICENSE).
