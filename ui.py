import flet as ft
import threading
import engine
import veille
import history
import queue
import asyncio
import time
import datetime
import os
from urllib.parse import urlparse

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")    # ressources de l'interface (icône athena.ico), quel que soit le dossier de lancement

# --- Palette : gris anthracite + accent orange (maquette validée) ---
BG        = "#2A2C30"   # fond de la fenêtre
LINE      = "#3A3D43"   # filets discrets
SURFACE   = "#33363B"   # barre de saisie, pastille d'état
SURFACE_2 = "#3F434A"   # boutons au repos
BORDER    = "#464A51"
PANEL     = "#2F3237"   # menu des réglages
TEXT      = "#ECEDEF"
TEXT_2    = "#A9AEB6"   # texte secondaire
ANSWER    = "#E3E5E8"   # texte des réponses
ACCENT    = "#F55A0D"
AGENDA_MINE   = "#F55A0D"            # pastille/heure des agendas à toi (orange d'Athena)
AGENDA_SHARED = "#6FA8DC"            # agendas partagés avec toi (bleu doux)
AGENDA_BG     = ("#303338", "#363A40")   # capsules des jours : deux gris très proches, un jour sur deux
VEILLE_FIT_COLORS = {"ok": "#8BC48A", "limit": "#E6C26A", "no": "#E08A8A", "unknown": "#7C828B"}      # tient / limite / trop gros / inconnu
VEILLE_FIT_LABELS = {"ok": "tient", "limit": "limite", "no": "trop gros", "unknown": "taille ?"}
MAIL_COLORS   = ("#6FA8DC", "#8BC48A", "#C9A0DC", "#E6C26A", "#E08A8A", "#7FC8C8")   # une teinte douce par boîte mail (pastille + heure)
ON_ACCENT = "#1B1C1F"   # texte sur fond orange (le blanc serait trop peu contrasté)
DOT_OFF   = "#7C828B"
CONTENT_W = 860         # largeur maximale de la colonne de conversation (elle se réduit si la fenêtre est étroite)
SIDEBAR_W = 280         # largeur de la barre latérale des discussions
DEFAULT_KEY = "__defaut__"   # valeur de la liste déroulante pour « Par défaut de Windows »

# Mots repérés dans les statuts envoyés par le moteur
ACTIVE_STATUS   = ("Écoute", "Transcription", "Réflexion", "Recherche", "Chargement")
THINKING_STATUS = ("Réflexion", "Recherche", "Lecture", "Routine")


def main(page: ft.Page):
    page.title = "Athena"
    page.bgcolor = BG
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.window.width = 1408                      # 720p (1280 x 720) agrandi de 10 % : 1408 x 792
    page.window.height = 792
    page.window.alignment = ft.Alignment.CENTER   # fenêtre centrée au milieu de l'écran au démarrage
    page.window.icon = "athena.ico"               # icône de la fenêtre et de la barre des tâches (Windows) : assets\athena.ico
    page.window.min_width = 920
    page.window.min_height = 560

    clipboard = ft.Clipboard()          # service du presse-papiers (bouton « Copier »)
    url_launcher = ft.UrlLauncher()     # ouvre les liens des sources dans le navigateur
    page.services.append(clipboard)
    page.services.append(url_launcher)

    assistant = None                    # le moteur, tant qu'il tourne
    mic_on = False                      # micro éteint (mode écrit) par défaut ; allumé = mode oral
    pending_sources = []                # liens reçus du moteur, rattachés à la prochaine réponse
    pending_veille = None               # News IA (cartes) demandées par le moteur, rattachée à la prochaine réponse
    pending_mails = None                # mails en capsules demandés par le moteur, rattachés à la prochaine réponse
    pending_agenda = None               # agenda en capsules demandé par le moteur, rattaché à la prochaine réponse
    pending_actions = []                # boutons d'action demandés par le moteur (ex. « thunderbird »), rattachés à la prochaine réponse
    # Audio choisi par l'utilisateur (mémorisé dans prefs.json) : la fenêtre « Audio » ne s'ouvre qu'au premier allumage du micro
    audio_pref = engine.load_prefs().get("audio")
    audio_pref = audio_pref if isinstance(audio_pref, dict) else {}
    audio_in = str(audio_pref.get("input") or "")           # "" = appareil par défaut de Windows
    audio_out = str(audio_pref.get("output") or "")
    audio_confirmed = bool(audio_pref.get("confirmed"))
    audio_done = None                                       # à appeler quand la fenêtre « Audio » est validée
    current_turn = None                # la question en cours (la réponse viendra s'y ajouter)
    pending = None                      # l'indicateur « Réflexion… / Recherche web… »
    pending_label = None
    ui_queue = queue.Queue()
    # Historique des discussions (history.json) : la discussion ouverte, la question en attente de sa réponse
    store = history.HistoryStore()
    if not engine.HISTORY_KEEP_PRIVATE:         # réglage [historique] garder_prive désactivé : on efface aussi ce qui avait été gardé avant
        n_scrubbed = store.scrub_private()
        if n_scrubbed: print(f"[Historique] garder_prive désactivé : {n_scrubbed} échange(s) privé(s) effacé(s) de l'historique.")
    conv_id = None                      # discussion ouverte (None = nouvelle, pas encore enregistrée)
    closing = False                     # la fermeture d'Athena est en cours (on ne la lance qu'une fois)
    routine_active = False              # une « routine du matin » est en cours (bouton Stop visible, blocs enregistrés un par un)
    last_question = None                # question posée, en attente de sa réponse pour être enregistrée
    turn_private = False                # la prochaine réponse est privée (mails, agenda) : enregistrée en entier seulement si garder_prive est vrai
    ui_prefs = engine.load_prefs().get("ui", {})
    sidebar_open = bool(ui_prefs.get("sidebar", True)) if isinstance(ui_prefs, dict) else True

    # ------------------------------------------------------------------ zone de saisie
    mic_btn = ft.Container(width=48, height=48, border_radius=24, alignment=ft.Alignment.CENTER, ink=True)
    text_input = ft.TextField(hint_text="Écris ta question…", hint_style=ft.TextStyle(color=TEXT_2), expand=True,
                              border=ft.NoInputBorder(), text_size=16, color=TEXT, cursor_color=ACCENT,
                              content_padding=ft.Padding.symmetric(horizontal=4, vertical=12), autofocus=True)
    send_btn = ft.Container(width=48, height=48, border_radius=24, alignment=ft.Alignment.CENTER, ink=True,
                            tooltip="Envoyer")
    hint = ft.Text("", size=13, color=TEXT_2, text_align=ft.TextAlign.CENTER)
    input_bar = ft.Container(width=CONTENT_W, bgcolor=SURFACE, border=ft.Border.all(1, BORDER), border_radius=32, padding=8,
                             content=ft.Row([mic_btn, text_input, send_btn], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER))
    stop_btn = ft.Container(visible=False, ink=True, border_radius=20, bgcolor=SURFACE, border=ft.Border.all(1, ACCENT),
                            padding=ft.Padding.symmetric(horizontal=18, vertical=8), tooltip="Couper la voix et arrêter la routine (ou tape « stop », ou dis « stop » entre deux parties)",
                            content=ft.Row([ft.Icon(ft.Icons.STOP_CIRCLE_OUTLINED, size=20, color=ACCENT),
                                            ft.Text("Stop", size=14, weight=ft.FontWeight.W_600, color=TEXT)], spacing=8, tight=True))
    footer = ft.Container(alignment=ft.Alignment.CENTER, padding=ft.Padding.only(left=24, right=24, top=8, bottom=18),
                          content=ft.Column([stop_btn, input_bar, hint], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8, tight=True))

    # ------------------------------------------------------------------ en-tête
    status_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor=DOT_OFF)
    status_label = ft.Text("Hors ligne", size=14, color=TEXT_2)
    status_pill = ft.Container(content=ft.Row([status_dot, status_label], spacing=8, tight=True), bgcolor=SURFACE,
                               border_radius=18, height=36, padding=ft.Padding.symmetric(horizontal=14), alignment=ft.Alignment.CENTER)
    settings_btn = ft.IconButton(icon=ft.Icons.SETTINGS, icon_color=TEXT_2, icon_size=22, tooltip="Réglages")
    run_btn = ft.Button(height=44)
    brand = ft.Row([ft.Container(width=12, height=12, border_radius=6, bgcolor=ACCENT),
                    ft.Text("Athena", size=18, weight=ft.FontWeight.W_600, color=TEXT)], spacing=10, expand=True)
    sidebar_btn = ft.IconButton(icon=ft.Icons.MENU, icon_color=TEXT_2, icon_size=22, tooltip="Afficher / masquer les discussions")
    # Pastille « News IA » (nouveaux modèles) : toujours visible à droite ; point orange avec le nombre de nouveautés ; clic = lance les News IA
    veille_count = ft.Container(visible=False, bgcolor=ACCENT, border_radius=9, height=18, alignment=ft.Alignment.CENTER,
                                padding=ft.Padding.symmetric(horizontal=6),
                                content=ft.Text("0", size=11, weight=ft.FontWeight.W_700, color=ON_ACCENT))
    veille_spin = ft.ProgressRing(width=14, height=14, stroke_width=2, color=ACCENT, visible=False)
    veille_pill = ft.Container(ink=True, bgcolor=SURFACE, border_radius=18, height=36, alignment=ft.Alignment.CENTER,
                               padding=ft.Padding.symmetric(horizontal=12), visible=veille.ACTIVE,
                               tooltip="News IA : clique pour afficher les nouveaux modèles",
                               content=ft.Row([ft.Icon(ft.Icons.NEWSPAPER, size=18, color=ACCENT), ft.Text("News IA", size=14, color=TEXT), veille_spin, veille_count],
                                              spacing=8, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER))
    header = ft.Container(padding=ft.Padding.symmetric(horizontal=24, vertical=12),
                          border=ft.Border.only(bottom=ft.BorderSide(1, LINE)),
                          content=ft.Row([sidebar_btn, brand, veille_pill, status_pill, settings_btn, run_btn], spacing=12,
                                         vertical_alignment=ft.CrossAxisAlignment.CENTER))

    # ------------------------------------------------------------------ corps : accueil ou conversation
    welcome_sub = ft.Text("", size=17, color=TEXT_2, text_align=ft.TextAlign.CENTER)
    welcome_wrap = ft.Container(expand=True, alignment=ft.Alignment.CENTER,
                                content=ft.Column([ft.Text("Athena", size=44, weight=ft.FontWeight.W_600, color=TEXT), welcome_sub],
                                                  horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12, tight=True))
    # La liste occupe TOUTE la largeur (de la barre latérale au bord droit) : la molette marche partout et la barre de défilement est au bord de la fenêtre.
    # La colonne de 860 px est centrée grâce aux marges gauche/droite calculées dans apply_layout.
    chat_list = ft.ListView(expand=True, spacing=32, padding=ft.Padding.only(left=24, right=24, top=36, bottom=24), auto_scroll=True)
    chat_col = ft.Column([chat_list], horizontal_alignment=ft.CrossAxisAlignment.STRETCH, expand=True, visible=False)
    body = ft.Column([welcome_wrap, chat_col], expand=True, spacing=0, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    # ------------------------------------------------------------------ menu des réglages (⚙)
    close_btn = ft.IconButton(icon=ft.Icons.CLOSE, icon_color=TEXT_2, icon_size=22, tooltip="Fermer")
    audio_in_text = ft.Text("", size=14, color=TEXT)
    audio_out_text = ft.Text("", size=14, color=TEXT)
    audio_note = ft.Text("", size=13, color=TEXT_2)
    show_tokens = bool(engine.load_prefs().get("ui", {}).get("tokens", True))
    tokens_switch = ft.Switch(value=show_tokens, active_color=ACCENT)
    audio_edit_btn = ft.Container(ink=True, border_radius=22, border=ft.Border.all(1, "#5A5F67"), height=44, alignment=ft.Alignment.CENTER,
                                  padding=ft.Padding.symmetric(horizontal=20),
                                  content=ft.Text("Modifier", size=15, weight=ft.FontWeight.W_600, color=TEXT))
    panel = ft.Container(
        right=0, top=0, bottom=0, width=360, visible=False, bgcolor=PANEL,
        border=ft.Border.only(left=ft.BorderSide(1, BORDER)), padding=ft.Padding.symmetric(horizontal=24, vertical=20),
        content=ft.Column([
            ft.Row([ft.Text("Réglages", size=20, weight=ft.FontWeight.W_600, color=TEXT), close_btn],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text("AUDIO", size=13, weight=ft.FontWeight.W_600, color=TEXT_2),
            ft.Container(bgcolor=SURFACE, border=ft.Border.all(1, BORDER), border_radius=14, padding=16,
                         content=ft.Column([ft.Text("Micro", size=12, color=TEXT_2), audio_in_text,
                                            ft.Text("Sortie audio", size=12, color=TEXT_2), audio_out_text], spacing=4)),
            ft.Row([audio_edit_btn]),
            audio_note,
            ft.Text("AFFICHAGE", size=13, weight=ft.FontWeight.W_600, color=TEXT_2),
            ft.Row([ft.Text("Compteur de tokens (en haut)", size=14, color=TEXT), tokens_switch],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text("La voix, la pause de fin de phrase, le modèle et les mots-clés de recherche se règlent dans config.toml.",
                    size=13, color=TEXT_2),
        ], spacing=20))
    scrim = ft.Container(left=0, top=0, right=0, bottom=0, visible=False, bgcolor=ft.Colors.with_opacity(0.6, "#121315"))

    # ------------------------------------------------------------------ encart « Génération » (en-tête, à droite du mot « Athena »)
    tok_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor=TEXT_2)
    tok_title = ft.Text("Génération", size=12, color=TEXT_2)
    tok_count = ft.Text("", size=12, weight=ft.FontWeight.W_600, color=TEXT)
    tok_detail = ft.Text("", size=12, color=TEXT_2)
    tokens_box = ft.Container(
        visible=False, bgcolor=SURFACE, border=ft.Border.all(1, BORDER), border_radius=12, margin=ft.Margin.only(left=14),
        padding=ft.Padding.symmetric(horizontal=12, vertical=6), tooltip="Tokens générés par l'IA (masquable dans ⚙)",
        content=ft.Column([ft.Row([tok_dot, tok_title, tok_count], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER), tok_detail],
                          spacing=2, tight=True))

    # ------------------------------------------------------------------ fenêtre « Audio » (choix du micro et de la sortie)
    dd_in = ft.Dropdown(label="Micro", width=412, text_size=14, border=ft.OutlineInputBorder(side=ft.BorderSide(1, BORDER)))
    dd_out = ft.Dropdown(label="Sortie audio", width=412, text_size=14, border=ft.OutlineInputBorder(side=ft.BorderSide(1, BORDER)))
    audio_cancel = ft.Container(ink=True, border_radius=22, height=44, alignment=ft.Alignment.CENTER, padding=ft.Padding.symmetric(horizontal=20),
                                content=ft.Text("Annuler", size=15, weight=ft.FontWeight.W_500, color=TEXT_2))
    audio_ok = ft.Button(height=44, content=ft.Text("Valider", size=15, weight=ft.FontWeight.W_600, color=ON_ACCENT),
                         style=ft.ButtonStyle(bgcolor=ACCENT, shape=ft.StadiumBorder(), padding=ft.Padding.symmetric(horizontal=24)))
    audio_modal = ft.Container(
        left=0, top=0, right=0, bottom=0, visible=False, alignment=ft.Alignment.CENTER,
        bgcolor=ft.Colors.with_opacity(0.6, "#121315"),
        content=ft.Container(width=460, bgcolor=PANEL, border=ft.Border.all(1, BORDER), border_radius=16, padding=24,
                             content=ft.Column([
                                 ft.Text("Audio", size=20, weight=ft.FontWeight.W_600, color=TEXT),
                                 ft.Text("Choisis le micro et la sortie audio pour parler avec Athena. "
                                         "Tu pourras les changer dans les réglages (⚙).", size=14, color=TEXT_2),
                                 dd_in, dd_out,
                                 ft.Row([audio_cancel, audio_ok], alignment=ft.MainAxisAlignment.END, spacing=8),
                             ], spacing=18, tight=True)))

    # ------------------------------------------------------------------ mises à jour de l'affichage
    def refresh_mic():
        mic_btn.bgcolor = ACCENT if mic_on else SURFACE_2
        mic_btn.content = ft.Icon(ft.Icons.MIC if mic_on else ft.Icons.MIC_OFF, size=22, color=ON_ACCENT if mic_on else "#C3C7CD")
        mic_btn.tooltip = "Micro allumé : cliquer pour l'éteindre" if mic_on else "Micro éteint : cliquer pour l'allumer"

    def refresh_send():
        has_text = bool((text_input.value or "").strip())
        send_btn.bgcolor = ACCENT if has_text else SURFACE_2
        send_btn.content = ft.Icon(ft.Icons.ARROW_UPWARD, size=22, color=ON_ACCENT if has_text else TEXT_2)

    def refresh_hint():
        if mic_on:
            hint.value = ("Micro allumé · Athena t'écoute et te répond à voix haute" if assistant
                          else "Micro allumé · Athena est arrêtée : clique sur « Relancer »")
        else:
            hint.value = "Micro éteint · mode écrit"

    def refresh_welcome():
        if assistant:
            welcome_sub.value = "Dis « Athena » ou écris ta question." if mic_on else "Écris ta question."
        else:
            welcome_sub.value = "Athena est arrêtée. Clique sur « Relancer » en haut à droite."

    def set_status(text):
        active = any(k in text for k in ACTIVE_STATUS)
        status_dot.bgcolor = ACCENT if active else DOT_OFF
        status_label.value = text.replace("...", "…")
        status_label.color = TEXT if active else TEXT_2

    def set_run_button(state):
        """Athena démarre toute seule : le bouton n'apparaît (« Relancer ») que si le moteur s'est arrêté (plantage, Ollama éteint...)."""
        run_btn.content = ft.Row([ft.Icon(ft.Icons.REFRESH, size=18, color=ON_ACCENT),
                                  ft.Text("Relancer", size=15, weight=ft.FontWeight.W_600, color=ON_ACCENT)], spacing=8, tight=True)
        run_btn.style = ft.ButtonStyle(bgcolor=ACCENT, shape=ft.StadiumBorder(), padding=ft.Padding.symmetric(horizontal=20))
        run_btn.disabled = False
        run_btn.visible = (state == "off")

    def refresh_audio_summary():
        audio_in_text.value = audio_in or "Par défaut de Windows"
        audio_out_text.value = audio_out or "Par défaut de Windows"
        audio_note.value = "" if audio_confirmed else "Pas encore choisi : la fenêtre s'ouvrira au premier allumage du micro."

    # ------------------------------------------------------------------ éléments de la conversation
    def show_chat():
        welcome_wrap.visible = False
        chat_col.visible = True

    def question_text(text):
        return ft.Text(text, size=20, weight=ft.FontWeight.W_600, color=TEXT, selectable=True)

    def source_card(number, src):
        url = src["url"]
        domain = urlparse(url).netloc.removeprefix("www.") or url

        async def open_link(e):
            await url_launcher.launch_url(url)

        def read_page(e):
            # « Lire » : Athena ouvre la page et la résume (le lien dans le message déclenche l'outil du moteur)
            if assistant:
                threading.Thread(target=assistant.process_text, args=(f"Lis et résume cette page : {url}",), daemon=True).start()

        read_btn = ft.Container(
            on_click=read_page, ink=True, border_radius=10, padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            tooltip="Athena lit la page et la résume",
            content=ft.Row([ft.Icon(ft.Icons.AUTO_STORIES, size=16, color=ACCENT),
                            ft.Text("Lire", size=13, weight=ft.FontWeight.W_500, color=TEXT_2)], spacing=6, tight=True))

        card = ft.Container(
            width=224, on_click=open_link, ink=True, border_radius=12, bgcolor=SURFACE, border=ft.Border.all(1, BORDER),
            padding=ft.Padding.symmetric(horizontal=14, vertical=12), tooltip=url,   # l'adresse complète s'affiche au survol
            content=ft.Column([
                ft.Row([ft.Container(width=18, height=18, border_radius=9, bgcolor=BORDER, alignment=ft.Alignment.CENTER,
                                     content=ft.Text(str(number), size=10, weight=ft.FontWeight.W_600, color=TEXT)),
                        ft.Text(domain, size=12, color=TEXT_2, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=8),
                ft.Text(src.get("title") or domain, size=14, weight=ft.FontWeight.W_500, color=TEXT, max_lines=2,
                        overflow=ft.TextOverflow.ELLIPSIS),
            ], spacing=8, tight=True))
        return ft.Column([card, read_btn], spacing=2, tight=True)

    def offer_card(number, src):
        """Offre d'emploi : carte pleine largeur (titre, lieu et site, extrait, boutons Ouvrir / Lire)."""
        url = src["url"]

        async def open_link(e):
            await url_launcher.launch_url(url)

        def read_page(e):
            if assistant:
                threading.Thread(target=assistant.process_text, args=(f"Lis et résume cette page : {url}",), daemon=True).start()

        def small_button(icon, label, on_click, strong=False):
            return ft.Container(
                on_click=on_click, ink=True, border_radius=10, padding=ft.Padding.symmetric(horizontal=12, vertical=7),
                bgcolor=ACCENT if strong else None, border=None if strong else ft.Border.all(1, BORDER),
                content=ft.Row([ft.Icon(icon, size=16, color=ON_ACCENT if strong else ACCENT),
                                ft.Text(label, size=13, weight=ft.FontWeight.W_600 if strong else ft.FontWeight.W_500,
                                        color=ON_ACCENT if strong else TEXT_2)], spacing=6, tight=True))

        body = [ft.Row([ft.Container(width=22, height=22, border_radius=11, bgcolor=BORDER, alignment=ft.Alignment.CENTER,
                                     content=ft.Text(str(number), size=11, weight=ft.FontWeight.W_600, color=TEXT)),
                        ft.Text(src.get("title") or url, size=16, weight=ft.FontWeight.W_600, color=TEXT, max_lines=2,
                                overflow=ft.TextOverflow.ELLIPSIS, expand=True)], spacing=10)]
        if src.get("subtitle"):
            body.append(ft.Row([ft.Icon(ft.Icons.PLACE_OUTLINED, size=15, color=ACCENT),
                                ft.Text(src["subtitle"], size=13, color=TEXT_2, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True)],
                               spacing=6))
        if src.get("snippet"):
            body.append(ft.Text(src["snippet"], size=14, color=ANSWER, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS))
        body.append(ft.Row([small_button(ft.Icons.OPEN_IN_NEW, "Ouvrir", open_link, strong=True),
                            small_button(ft.Icons.AUTO_STORIES, "Lire", read_page)], spacing=8))
        return ft.Container(border_radius=12, bgcolor=SURFACE, border=ft.Border.all(1, BORDER), tooltip=url,
                            padding=ft.Padding.symmetric(horizontal=16, vertical=14), content=ft.Column(body, spacing=8, tight=True))

    def sources_block(sources):
        if sources and all(s.get("kind") == "offre" for s in sources):
            return ft.Column([
                ft.Text("OFFRES", size=13, weight=ft.FontWeight.W_600, color=TEXT_2),
                ft.Column([offer_card(i, s) for i, s in enumerate(sources, 1)], spacing=10),
            ], spacing=10)
        return ft.Column([
            ft.Text("SOURCES", size=13, weight=ft.FontWeight.W_600, color=TEXT_2),
            ft.Row([source_card(i, s) for i, s in enumerate(sources, 1)], spacing=12, run_spacing=12, wrap=True),
        ], spacing=10)

    def thunderbird_button():
        """Bouton « Ouvrir Thunderbird » avec l'icône de ton Thunderbird (sinon une enveloppe)."""
        def open_mail(e):
            def work():
                try:
                    engine.open_thunderbird()
                except engine.MailError as ex:
                    flash_hint(str(ex))
            threading.Thread(target=work, daemon=True).start()

        icon = (ft.Image(src=engine.THUNDERBIRD_ICON, width=22, height=22) if os.path.isfile(engine.THUNDERBIRD_ICON)
                else ft.Icon(ft.Icons.MAIL_OUTLINE, size=20, color=ACCENT))
        return ft.Container(on_click=open_mail, ink=True, border_radius=12, border=ft.Border.all(1, BORDER), bgcolor=SURFACE,
                            padding=ft.Padding.symmetric(horizontal=14, vertical=9), tooltip="Ouvrir Thunderbird sur ta boîte de réception",
                            content=ft.Row([icon, ft.Text("Ouvrir Thunderbird", size=14, weight=ft.FontWeight.W_500, color=TEXT)],
                                           spacing=10, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER))

    def veille_cards(data):
        """News IA en capsules : « Modèles » (pastille verte = tient, jaune = limite, rouge = trop gros) puis « Articles » (en français)."""
        def open_url(url):
            async def go(e): await url_launcher.launch_url(url)
            return go

        def read_url(url):
            def go(e):
                if assistant: threading.Thread(target=assistant.process_text, args=(f"Lis et résume cette page : {url}",), daemon=True).start()
            return go

        def small_button(icon, label, on_click, strong=False):
            return ft.Container(on_click=on_click, ink=True, border_radius=10, padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                                bgcolor=ACCENT if strong else None, border=None if strong else ft.Border.all(1, BORDER),
                                content=ft.Row([ft.Icon(icon, size=15, color=ON_ACCENT if strong else ACCENT),
                                                ft.Text(label, size=13, weight=ft.FontWeight.W_600 if strong else ft.FontWeight.W_500,
                                                        color=ON_ACCENT if strong else TEXT_2)], spacing=6, tight=True))

        def new_tag():
            return ft.Container(bgcolor=ACCENT, border_radius=8, padding=ft.Padding.symmetric(horizontal=7, vertical=1),
                                content=ft.Text("NOUVEAU", size=10, weight=ft.FontWeight.W_700, color=ON_ACCENT))

        def model_row(m):
            color = VEILLE_FIT_COLORS.get(m["fit"], DOT_OFF)
            size = "taille inconnue" if m["size_gb"] is None else f"{'≈ ' if m['estimated'] else ''}{m['size_gb']:.1f} Go"
            head = [ft.Text(m["name"], size=15, weight=ft.FontWeight.W_600, color=TEXT, selectable=True)]
            if m["star"]: head.insert(0, ft.Icon(ft.Icons.STAR, size=16, color=ACCENT))
            if m["new"]: head.append(new_tag())
            meta = [ft.Container(width=9, height=9, border_radius=5, bgcolor=color),
                    ft.Text(f"{VEILLE_FIT_LABELS.get(m['fit'], '')} · {size}", size=12, weight=ft.FontWeight.W_600, color=color)]
            extra = " · ".join(x for x in (", ".join(m["caps"]), m["by"], m["date"]) if x)
            meta.append(ft.Text(extra, size=12, color=TEXT_2))
            body = [ft.Row(head, spacing=8, wrap=True, run_spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Row(meta, spacing=8, wrap=True, run_spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)]
            if m.get("desc"): body.append(ft.Text(m["desc"], size=13, color=ANSWER, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
            if m.get("pull"): body.append(ft.Text(m["pull"], size=12, color=TEXT_2, selectable=True, font_family="Consolas"))
            body.append(ft.Row([small_button(ft.Icons.OPEN_IN_NEW, "Ouvrir", open_url(m["url"]), strong=True),
                                small_button(ft.Icons.AUTO_STORIES, "Lire", read_url(m["url"]))], spacing=8))
            return ft.Column(body, spacing=5)

        def article_row(a):
            head = [ft.Text(a["title"], size=15, weight=ft.FontWeight.W_600, color=TEXT, selectable=True)]
            if a["new"]: head.append(new_tag())
            return ft.Column([ft.Row(head, spacing=8, wrap=True, run_spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                              ft.Text(" · ".join(x for x in (a["source"], a["date"]) if x), size=12, color=TEXT_2),
                              ft.Row([small_button(ft.Icons.OPEN_IN_NEW, "Ouvrir", open_url(a["url"]), strong=True),
                                      small_button(ft.Icons.AUTO_STORIES, "Lire", read_url(a["url"]))], spacing=8)], spacing=5)

        out = []
        if data.get("recap"):
            out.append(ft.Text("Rien de nouveau depuis tes dernières news IA. Derniers éléments connus :", size=14, color=TEXT_2))
        for i, (title, rows) in enumerate((("Modèles", [model_row(m) for m in data.get("models") or []]),
                                           ("Articles (en français)", [article_row(a) for a in data.get("articles") or []]))):
            if rows:
                out.append(ft.Container(bgcolor=AGENDA_BG[i % 2], border_radius=16, padding=ft.Padding.symmetric(horizontal=16, vertical=12),
                                        content=ft.Column([ft.Text(title, size=15, weight=ft.FontWeight.W_600, color=TEXT)] + rows, spacing=14)))
        notes = []
        if data.get("hidden_big"):
            notes.append(f"{data['hidden_big']} modèle(s) trop gros pour ta carte (plus de {veille.SIZE_LIMIT_GB:g} Go) non affichés.")
        notes += [f"⚠ Source indisponible — {e}" for e in data.get("errors") or []]
        out += [ft.Text(n, size=13, color=TEXT_2, selectable=True) for n in notes]
        return ft.Column(out, spacing=10)

    def mail_cards(data):
        """Mails en capsules : une par boîte (fond légèrement différent d'une boîte à l'autre, pastille d'une couleur par boîte).
        Avec la liste : heure + objet, puis expéditeur et adresse en petit. Sans liste (routine) : seulement le nombre."""
        out = []
        for i, box in enumerate(data["boxes"]):
            color = MAIL_COLORS[i % len(MAIL_COLORS)]
            n, shown = box["total"], len(box["mails"])
            count = f"{n} non lu{'s' if n > 1 else ''}" + (f" (les {shown} plus récents)" if data.get("listing") and shown < n else "")
            body = [ft.Row([ft.Container(width=9, height=9, border_radius=5, bgcolor=color),
                            ft.Text(box["name"], size=15, weight=ft.FontWeight.W_600, color=TEXT),
                            ft.Text(count, size=13, color=TEXT_2)], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)]
            for m in box["mails"]:
                sender = "  ·  ".join(x for x in (m["name"], m["addr"]) if x)
                lines = [ft.Row([ft.Container(width=86, content=ft.Text(m["when"], size=13, weight=ft.FontWeight.W_600, color=color)),
                                 ft.Text(m["subject"], size=15, color=ANSWER, selectable=True, expand=True)],
                                vertical_alignment=ft.CrossAxisAlignment.START, spacing=8)]
                if sender: lines.append(ft.Row([ft.Container(width=94), ft.Text(sender, size=12, color=TEXT_2, selectable=True, expand=True)], spacing=0))
                body.append(ft.Column(lines, spacing=2))
            out.append(ft.Container(bgcolor=AGENDA_BG[i % 2], border_radius=16, padding=ft.Padding.symmetric(horizontal=16, vertical=12),
                                    content=ft.Column(body, spacing=10)))
        for line in data.get("errors") or []:
            out.append(ft.Text(line, size=14, color=TEXT_2, selectable=True))
        return ft.Column(out, spacing=10)

    def agenda_connect_button():
        """Bouton « Renouveler l'accès à l'agenda » : lance Connecter-agenda.bat (fenêtre à part + navigateur)."""
        def run_bat(e):
            def work():
                try:
                    engine.launch_agenda_connect()
                    flash_hint("Fenêtre de connexion ouverte : accepte dans ton navigateur, puis redemande ton agenda.")
                except engine.agenda.AgendaError as ex:
                    flash_hint(str(ex))
            threading.Thread(target=work, daemon=True).start()

        return ft.Container(on_click=run_bat, ink=True, border_radius=12, border=ft.Border.all(1, BORDER), bgcolor=SURFACE,
                            padding=ft.Padding.symmetric(horizontal=14, vertical=9), tooltip="Lance Connecter-agenda.bat pour te reconnecter à Google Agenda",
                            content=ft.Row([ft.Icon(ft.Icons.EVENT_REPEAT, size=20, color=ACCENT),
                                            ft.Text("Renouveler l'accès à l'agenda", size=14, weight=ft.FontWeight.W_500, color=TEXT)],
                                           spacing=10, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER))

    def agenda_cards(data):
        """Agenda en capsules : une par jour (fond légèrement différent un jour sur deux), pastille orange = tes agendas, bleue = partagés."""
        def dot(mine): return ft.Container(width=9, height=9, border_radius=5, bgcolor=AGENDA_MINE if mine else AGENDA_SHARED, margin=ft.Margin.only(top=6))

        def event_row(ev, show_cal):
            color = AGENDA_MINE if ev["mine"] else AGENDA_SHARED
            detail = [x for x in (ev["location"], ev["calendar"] if show_cal else "") if x]
            lines = [ft.Row([ft.Container(width=118, content=ft.Text(ev["when"], size=14, weight=ft.FontWeight.W_600, color=color)),
                             ft.Text(ev["title"], size=15, color=ANSWER, selectable=True, expand=True)],
                            vertical_alignment=ft.CrossAxisAlignment.START, spacing=8)]
            if detail:
                lines.append(ft.Text("  ·  ".join(detail), size=12, color=TEXT_2, selectable=True))
            return ft.Row([dot(ev["mine"]), ft.Column(lines, spacing=2, expand=True)], vertical_alignment=ft.CrossAxisAlignment.START, spacing=10)

        out = []
        for i, day in enumerate(data["days"]):
            body = [ft.Text(day["title"], size=15, weight=ft.FontWeight.W_600, color=TEXT)]
            body += [event_row(ev, data["show_calendar"]) for ev in day["events"]]
            out.append(ft.Container(bgcolor=AGENDA_BG[i % 2], border_radius=16, padding=ft.Padding.symmetric(horizontal=16, vertical=12),
                                    content=ft.Column(body, spacing=9)))
        for line in [data.get("empty"), data.get("more"), data.get("count")] + list(data.get("errors") or []):
            if line: out.append(ft.Text(line, size=14, color=TEXT_2, selectable=True))
        if data.get("mixed"):
            out.append(ft.Row([ft.Container(width=9, height=9, border_radius=5, bgcolor=AGENDA_MINE), ft.Text("Mes agendas", size=12, color=TEXT_2),
                               ft.Container(width=9, height=9, border_radius=5, bgcolor=AGENDA_SHARED), ft.Text("Partagés avec moi", size=12, color=TEXT_2)],
                              spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER))
        return ft.Column(out, spacing=10)

    def ai_block(text, sources=None, actions=None, agenda=None, mails=None, veille_data=None):
        copy_icon = ft.Icon(ft.Icons.CONTENT_COPY, size=18, color=TEXT_2)
        copy_label = ft.Text("Copier", size=14, color=TEXT_2)

        async def copy(e):
            await clipboard.set(text)
            copy_icon.icon, copy_label.value = ft.Icons.CHECK, "Copié"
            page.update()
            await asyncio.sleep(1.5)
            copy_icon.icon, copy_label.value = ft.Icons.CONTENT_COPY, "Copier"
            page.update()

        parts = [
            ft.Row([ft.Container(width=10, height=10, border_radius=5, bgcolor=ACCENT),
                    ft.Text("Athena", size=14, weight=ft.FontWeight.W_600, color=TEXT)], spacing=8),
            (ft.Column([ft.Text(text.partition("\n\n")[0], size=17, color=ANSWER, selectable=True), agenda_cards(agenda)], spacing=12)
             if agenda else
             ft.Column([ft.Text(text, size=17, color=ANSWER, selectable=True), mail_cards(mails)], spacing=12) if mails
             else ft.Column([ft.Text(text.partition("\n\n")[0], size=17, color=ANSWER, selectable=True), veille_cards(veille_data)], spacing=12) if veille_data
             else ft.Text(text, size=17, color=ANSWER, selectable=True)),
            ft.Row([ft.Container(on_click=copy, ink=True, border_radius=10, padding=ft.Padding.symmetric(horizontal=10, vertical=10),
                                 content=ft.Row([copy_icon, copy_label], spacing=8, tight=True))]),
        ]
        if "thunderbird" in (actions or []):
            parts.append(ft.Row([thunderbird_button()]))       # à la fin du message
        if "agenda_connect" in (actions or []):
            parts.append(ft.Row([agenda_connect_button()]))
        if sources:
            parts.append(sources_block(sources))
        return ft.Column(parts, spacing=14)

    def system_note(text):
        return ft.Container(alignment=ft.Alignment.CENTER, content=ft.Text(text, size=13, color=TEXT_2))

    def show_pending(label):
        nonlocal pending, pending_label
        if pending is None:
            pending_label = ft.Text(label, size=15, color=TEXT_2)
            pending = ft.Row([ft.ProgressRing(width=16, height=16, stroke_width=2, color=ACCENT), pending_label], spacing=12)
            current_turn.controls.append(pending)
        else:
            pending_label.value = label

    def hide_pending():
        nonlocal pending, pending_label
        if pending is not None and current_turn is not None and pending in current_turn.controls:
            current_turn.controls.remove(pending)
        pending = pending_label = None

    # ------------------------------------------------------------------ barre latérale : historique des discussions
    side_list = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, expand=True)
    new_chat_btn = ft.Container(ink=True, border_radius=12, border=ft.Border.all(1, BORDER), padding=ft.Padding.symmetric(horizontal=14, vertical=10),
                                tooltip="Commencer une nouvelle discussion",
                                content=ft.Row([ft.Icon(ft.Icons.ADD, size=18, color=ACCENT),
                                                ft.Text("Nouvelle discussion", size=14, weight=ft.FontWeight.W_500, color=TEXT)], spacing=8, tight=True))
    sidebar = ft.Container(
        width=SIDEBAR_W, visible=sidebar_open, bgcolor=PANEL, border=ft.Border.only(right=ft.BorderSide(1, LINE)),
        padding=ft.Padding.only(left=14, right=10, top=16, bottom=14),
        content=ft.Column([
            ft.Text("DISCUSSIONS", size=13, weight=ft.FontWeight.W_600, color=TEXT_2),
            new_chat_btn,
            side_list,
            ft.Text(f"Les discussions non épinglées sont supprimées après {history.RETENTION_DAYS} jours sans activité. "
                    "Épingle-les pour les garder.", size=12, color=TEXT_2),
        ], spacing=12, expand=True))

    def when(ts):
        """« Aujourd'hui 14:05 », « Hier », « Il y a 2 jours »."""
        d, today = datetime.datetime.fromtimestamp(ts), datetime.date.today()
        days = (today - d.date()).days
        if days <= 0: return f"Aujourd'hui {d:%H:%M}"
        return "Hier" if days == 1 else f"Il y a {days} jours"

    def is_busy():
        return current_turn is not None or pending is not None

    def refresh_sidebar():
        items = store.items(keep=conv_id)
        rows = []

        def section(label, group):
            if not group: return
            rows.append(ft.Text(label, size=12, weight=ft.FontWeight.W_600, color=TEXT_2))
            rows.extend(conv_row(it) for it in group)

        def conv_row(it):
            cid, active = it["id"], it["id"] == conv_id
            pin_btn = ft.IconButton(icon=ft.Icons.PUSH_PIN if it["pinned"] else ft.Icons.PUSH_PIN_OUTLINED, icon_size=16,
                                    icon_color=ACCENT if it["pinned"] else TEXT_2, tooltip="Désépingler" if it["pinned"] else "Épingler (garder)",
                                    on_click=lambda e, cid=cid: toggle_pin(cid))
            del_btn = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, icon_size=16, icon_color=TEXT_2, tooltip="Supprimer cette discussion",
                                    on_click=lambda e, cid=cid: delete_conversation(cid))
            return ft.Container(
                ink=True, border_radius=10, bgcolor=SURFACE if active else None, padding=ft.Padding.only(left=10, top=2, bottom=2),
                on_click=lambda e, cid=cid: open_conversation(cid), tooltip=it["title"],
                content=ft.Row([ft.Column([ft.Text(it["title"], size=14, color=TEXT, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                                           ft.Text("Épinglée" if it["pinned"] else when(it["updated"]), size=12, color=TEXT_2)],
                                          spacing=1, tight=True, expand=True), pin_btn, del_btn], spacing=0,
                               vertical_alignment=ft.CrossAxisAlignment.CENTER))

        section("ÉPINGLÉES", [it for it in items if it["pinned"]])
        section("RÉCENTES", [it for it in items if not it["pinned"]])
        if not items:
            rows.append(ft.Text("Aucune discussion pour l'instant.", size=13, color=TEXT_2))
        side_list.controls = rows

    async def flash_hint_async(msg):
        hint.value, hint.color = msg, ACCENT
        page.update()
        await asyncio.sleep(2.5)
        hint.color = TEXT_2
        refresh_hint()
        page.update()

    def flash_hint(msg):
        async def run():
            await flash_hint_async(msg)
        page.run_task(run)

    def engine_turns(cid):
        """Échanges d'une discussion pour redonner son contexte au modèle (les échanges privés — mails, agenda — n'y sont JAMAIS)."""
        c = store.get(cid)
        return [t for t in (c["turns"] if c else []) if not history.is_private(t)]

    # Forme attendue des données d'affichage (reçues du moteur, ou relues dans history.json) : on ne dessine que ce qui est bien formé
    def agenda_ok(d):
        return isinstance(d, dict) and isinstance(d.get("days"), list)

    def mails_ok(d):
        return isinstance(d, dict) and isinstance(d.get("boxes"), list) and all(isinstance(b, dict) and isinstance(b.get("mails"), list) for b in d["boxes"])

    def veille_ok(d):
        return isinstance(d, dict) and isinstance(d.get("models"), list) and isinstance(d.get("articles"), list)

    def render_turn(t):
        """Un échange de l'historique redessiné COMME À L'ORIGINE : texte, capsules de mails / agenda, cartes News IA, boutons, sources.
        Un bloc de la routine du matin garde son titre dans le bloc (pas de bulle « question »). Si les données sont abîmées, on retombe
        sur le texte et les sources seuls."""
        try:
            extra = t.get("extra") or {}
            text = (t["part"] + "\n\n" + t["a"]) if t.get("part") else t["a"]
            block = ai_block(text, t["sources"], extra.get("actions"),
                             extra.get("agenda") if agenda_ok(extra.get("agenda")) else None,
                             extra.get("mails") if mails_ok(extra.get("mails")) else None,
                             extra.get("veille") if veille_ok(extra.get("veille")) else None)
            return ft.Column([block] if t.get("part") else [question_text(t["q"]), block], spacing=16)
        except Exception as ex:
            print(f"[Historique] un échange n'a pas pu être redessiné à l'identique ({type(ex).__name__}) : texte seul affiché.")
            return ft.Column([question_text(t["q"]), ai_block(t["a"], t["sources"])], spacing=16)

    def show_welcome():
        chat_list.controls.clear()
        chat_col.visible = False
        welcome_wrap.visible = True

    async def scroll_to_end():
        """Descend tout en bas, cartes Sources comprises : leur hauteur n'est connue qu'après l'affichage, donc on recommence
        un peu plus tard (le défilement automatique seul s'arrêtait au bas de la réponse)."""
        try:
            chat_list.auto_scroll = False            # scroll_to n'agit que si le défilement automatique est coupé
            page.update()
            for delay in (0.1, 0.4):
                await asyncio.sleep(delay)
                await chat_list.scroll_to(offset=-1, duration=0)
        except Exception as ex:
            print(f"[UI] défilement impossible : {ex}")
        finally:
            chat_list.auto_scroll = True             # les nouvelles réponses continuent de suivre le bas
            page.update()

    def open_conversation(cid):
        nonlocal conv_id, last_question
        if is_busy():
            flash_hint("Attends la fin de la réponse avant de changer de discussion.")
            return
        c = store.get(cid)
        if c is None:
            refresh_sidebar(); page.update()
            return
        chat_list.controls.clear()
        for t in c["turns"]:
            chat_list.controls.append(render_turn(t))
        conv_id, last_question = cid, None
        show_chat()
        if assistant: assistant.restore_history(engine_turns(cid))      # le modèle retrouve le contexte
        refresh_sidebar()
        page.update()
        page.run_task(scroll_to_end)                                    # en bas de la discussion, sources visibles

    async def new_conversation(e=None):
        nonlocal conv_id, last_question
        if is_busy():
            flash_hint("Attends la fin de la réponse avant de changer de discussion.")
            return
        conv_id, last_question = None, None
        show_welcome()
        if assistant: assistant.reset_conversation()                    # le modèle oublie la discussion précédente
        refresh_sidebar()
        page.update()
        await text_input.focus()

    def toggle_pin(cid):
        c = store.get(cid)
        if c is not None: store.pin(cid, not c["pinned"])
        refresh_sidebar()
        page.update()

    def delete_conversation(cid):
        if cid == conv_id and is_busy():
            flash_hint("Attends la fin de la réponse avant de supprimer cette discussion.")
            return
        store.delete(cid)
        if cid == conv_id:
            page.run_task(new_conversation)
        else:
            refresh_sidebar()
            page.update()

    def set_veille_badge(count=None, state=None):
        """Met à jour la pastille : `count` = nouveautés (None = inchangé), `state` = "checking" (vérification en cours) ou "error"."""
        veille_spin.visible = state == "checking"
        if count is not None:
            veille_count.visible = count > 0
            veille_count.content.value = str(count)
        n = int(veille_count.content.value or 0) if veille_count.visible else 0
        veille_pill.tooltip = ("Vérification des nouveautés…" if state == "checking"
                               else "News IA indisponibles pour l'instant : clique pour réessayer" if state == "error"
                               else f"{n} nouveauté(s) à voir : clique pour les afficher" if n
                               else "News IA : clique pour afficher les nouveaux modèles")

    def veille_click(e):
        if not assistant:
            flash_hint("Athena est arrêtée : clique sur « Relancer ».")
        elif is_busy():
            flash_hint("Attends la fin de la réponse avant de lancer les News IA.")
        else:
            threading.Thread(target=assistant.process_text, args=("donne-moi les news IA",), daemon=True).start()     # même chemin que la commande vocale

    def veille_startup_check():
        """Vérification discrète au démarrage (en arrière-plan) : seulement le compteur de la pastille, rien n'est affiché ni marqué comme vu."""
        ui_queue.put(("veille_state", "checking"))
        try:
            veille.get_data(force=True)
            ui_queue.put(("veille_badge", veille.pending_count()))
        except Exception:
            ui_queue.put(("veille_state", "error"))

    def apply_layout(e=None):
        """Adapte la largeur de la conversation à la fenêtre (avec ou sans barre latérale)."""
        w = getattr(page, "width", None) or getattr(page.window, "width", None)     # largeur réelle, sinon celle demandée au démarrage
        if not w: return
        avail = int(w - (SIDEBAR_W if sidebar.visible else 0))                 # largeur de la zone à droite de la barre latérale
        cw = max(420, min(CONTENT_W, avail - 48))                              # largeur de la colonne de conversation
        side = max(24, (avail - cw) // 2)                                      # marge de chaque côté : la colonne reste centrée
        chat_list.padding = ft.Padding.only(left=side, right=side, top=36, bottom=24)
        input_bar.width = cw

    def on_resize(e):
        apply_layout()
        page.update()

    def toggle_sidebar(e):
        nonlocal sidebar_open
        sidebar_open = not sidebar_open
        sidebar.visible = sidebar_open
        prefs = engine.load_prefs()
        engine.save_prefs({**prefs, "ui": {**(prefs.get("ui") if isinstance(prefs.get("ui"), dict) else {}), "sidebar": sidebar_open}})
        apply_layout()
        page.update()

    # ------------------------------------------------------------------ messages venant du moteur
    def handle_ui_message(msg_type, data):
        nonlocal assistant, current_turn, pending_sources, pending_actions, pending_agenda, pending_mails, pending_veille, mic_on, conv_id, last_question, turn_private, routine_active
        if msg_type == "mode":
            # Le moteur a changé de mode de lui-même (ex. chargement de la voix impossible) : le micro suit
            mic_on = (data == "oral")
            refresh_mic()
            refresh_hint()
            refresh_welcome()
        elif msg_type == "tokens":
            running = data.get("state") == "run"
            tok_dot.bgcolor = ACCENT if running else TEXT_2
            tok_count.value = f"{data.get('tokens', 0)} tokens"
            if running:
                tok_detail.value = f"{data.get('elapsed', 0):.1f} s"
            else:
                tok_detail.value = f"{data.get('tps', 0):.1f} tokens/s  ·  {data.get('elapsed', 0):.1f} s"
            tokens_box.visible = show_tokens
        elif msg_type == "status":
            set_status(data)
            thinking = any(k in data for k in THINKING_STATUS)
            if routine_active and thinking and current_turn is None:      # entre deux parties de la routine : on garde l'indicateur d'attente
                current_turn = ft.Column([], spacing=16)
                chat_list.controls.append(current_turn)
            if current_turn is not None and thinking:
                show_pending(data.replace("...", "…"))
            else:
                hide_pending()
        elif msg_type == "routine":
            routine_active = bool(data)
            stop_btn.visible = routine_active                            # bouton « Stop » visible pendant la routine
            if not routine_active:
                hide_pending()
                current_turn = None
        elif msg_type == "user_text":
            hide_pending()
            show_chat()
            pending_sources, pending_actions, pending_agenda, pending_mails, pending_veille = [], [], None, None, None
            last_question, turn_private = data, False
            current_turn = ft.Column([question_text(data)], spacing=16)
            chat_list.controls.append(current_turn)
        elif msg_type == "quit":
            page.run_task(shutdown)          # commande vocale « Déconnecte » : on ferme tout, comme la croix de la fenêtre
        elif msg_type == "action":
            if data in ("thunderbird", "agenda_connect"): pending_actions.append(data)      # liste blanche : seul un bouton connu peut apparaître
        elif msg_type == "agenda":
            pending_agenda = data if agenda_ok(data) else None         # données de l'agenda (capsules)
        elif msg_type == "veille":
            ok_shape = veille_ok(data)
            pending_veille = data if ok_shape else None                # cartes de la veille
            if ok_shape: set_veille_badge(0)                           # ce qui s'affiche maintenant n'est plus « nouveau »
        elif msg_type == "veille_badge":
            set_veille_badge(int(data or 0))
        elif msg_type == "veille_state":
            set_veille_badge(state=data)
        elif msg_type == "mails":
            pending_mails = data if mails_ok(data) else None           # données des mails (capsules)
        elif msg_type == "private":
            turn_private = True              # mails / agenda : écrits en entier dans history.json seulement si garder_prive est vrai (sinon note neutre)
        elif msg_type == "sources":
            # Valeur reçue du moteur : on ne garde que des liens http(s) bien formés
            pending_sources = [s for s in data if isinstance(s, dict) and str(s.get("url", "")).lower().startswith(("http://", "https://"))]
        elif msg_type == "ai_text":
            hide_pending()
            show_chat()
            if current_turn is None:        # réponse sans question (erreur, veille...)
                current_turn = ft.Column([], spacing=16)
                chat_list.controls.append(current_turn)
            current_turn.controls.append(ai_block(data, pending_sources, pending_actions, pending_agenda, pending_mails, pending_veille))
            extra = {"actions": pending_actions, "agenda": pending_agenda, "mails": pending_mails, "veille": pending_veille}      # de quoi redessiner ce bloc à l'identique
            pending_actions, pending_agenda, pending_mails, pending_veille = [], None, None, None
            question, saved, part = last_question, data, None
            if question is None and routine_active:        # parties suivantes de la routine : le titre du bloc sert de question
                title, sep, rest = data.partition("\n\n")
                question, saved = title.strip()[:80], (rest or data)
                if sep and rest: part = title              # le titre fait partie du bloc : à la réouverture, pas de bulle « question »
            if question is not None:         # un échange complet (question + réponse) : on l'enregistre dans l'historique
                conv_id = store.add_turn(conv_id, question, saved, pending_sources, private=turn_private, extra=extra, part=part,
                                         keep_private=engine.HISTORY_KEEP_PRIVATE)
                refresh_sidebar()
            last_question, turn_private = None, False
            pending_sources = []
            current_turn = None
        elif msg_type == "stop":
            # Le moteur est arrêté (bouton, commande vocale ou erreur) : on remet l'interface à zéro
            hide_pending()
            current_turn = None
            routine_active = False
            stop_btn.visible = False
            last_question, turn_private = None, False
            assistant = None
            set_run_button("off")
            set_status("Hors ligne")
            refresh_hint()
            refresh_welcome()
            if chat_col.visible:
                chat_list.controls.append(system_note("Système arrêté."))

    async def process_ui_queue():
        while True:
            changed = False
            # On traite tous les messages en attente, puis on rafraîchit une seule fois
            while True:
                try:
                    msg_type, data = ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    handle_ui_message(msg_type, data)
                    changed = True
                except Exception as ex:
                    print(f"[UI] erreur sur le message {msg_type} : {ex}")
            if changed:
                try:
                    page.update()
                except RuntimeError:         # la fenêtre vient d'être fermée (shutdown) : plus rien à afficher, on arrête proprement
                    return
            await asyncio.sleep(0.1)

    callbacks = {"on_status": lambda text: ui_queue.put(("status", text)), "on_user_text": lambda text: ui_queue.put(("user_text", text)),
                 "on_ai_text": lambda text: ui_queue.put(("ai_text", text)), "on_sources": lambda sources: ui_queue.put(("sources", sources)), "on_stop": lambda: ui_queue.put(("stop", "")),
                 "on_mode": lambda mode: ui_queue.put(("mode", mode)),
                 "on_private": lambda: ui_queue.put(("private", "")),
                 "on_routine": lambda active: ui_queue.put(("routine", bool(active))),
                 "on_action": lambda name: ui_queue.put(("action", name)), "on_agenda": lambda data: ui_queue.put(("agenda", data)), "on_mails": lambda data: ui_queue.put(("mails", data)), "on_veille": lambda data: ui_queue.put(("veille", data)),
                 "on_quit": lambda: ui_queue.put(("quit", "")),
                 "on_tokens": lambda info: ui_queue.put(("tokens", info)),
                 "on_log": lambda msg: print(msg)}

    # ------------------------------------------------------------------ actions
    def start_ia():
        nonlocal assistant
        assistant = engine.AssistantEngine(callbacks)
        if conv_id: assistant.restore_history(engine_turns(conv_id))     # une discussion est déjà ouverte à l'écran : on lui redonne son contexte
        set_run_button("on")
        set_status("Chargement des modèles...")
        refresh_hint()
        refresh_welcome()
        page.update()
        # Mode texte par défaut : la voix ne se charge que si le micro est déjà allumé au démarrage
        threading.Thread(target=assistant.start, args=(mic_on, audio_in or None, audio_out or None), daemon=True).start()

    def on_run_click(e):
        if assistant is None:               # « Relancer » : le moteur s'était arrêté
            start_ia()

    async def shutdown():
        """Ferme Athena proprement : arrête le moteur, libère la VRAM (décharge le modèle d'Ollama), puis ferme la fenêtre."""
        nonlocal assistant, closing
        if closing: return
        closing = True
        set_status("Fermeture...")
        page.update()
        a, assistant = assistant, None
        if a: await asyncio.to_thread(a.stop)
        await asyncio.to_thread(engine.unload_model, print)       # la VRAM est libérée tout de suite, sans attendre les 30 min
        await page.window.destroy()

    def on_window_event(e):
        if e.type == ft.WindowEventType.CLOSE:        # la croix de la fenêtre (prevent_close est actif : c'est nous qui fermons)
            page.run_task(shutdown)

    async def set_mic(on):
        """Allume ou éteint le micro. Allumer charge la voix la première fois (quelques secondes, dans un thread)."""
        nonlocal mic_on
        mic_on = on
        refresh_mic()
        refresh_hint()
        refresh_welcome()
        if assistant:
            if on:
                threading.Thread(target=assistant.enable_oral, args=(audio_in or None, audio_out or None), daemon=True).start()
            else:
                assistant.disable_oral()
                set_status("Mode écrit activé")
        page.update()
        if not on:                        # on peut taper tout de suite, sans cliquer dans le champ
            await text_input.focus()

    async def toggle_mic(e):
        if mic_on:
            await set_mic(False)
        elif audio_confirmed:
            await set_mic(True)
        else:                             # première fois : on demande d'abord quel micro et quelle sortie utiliser
            open_audio_dialog(lambda: set_mic(True))

    # --- fenêtre « Audio »
    def open_audio_dialog(on_done):
        nonlocal audio_done
        devices = engine.list_audio_devices()

        def options(names):
            return [ft.dropdown.Option(key=DEFAULT_KEY, text="Par défaut de Windows")] + [ft.dropdown.Option(key=n, text=n) for n in names]

        dd_in.options = options(devices["input"])
        dd_out.options = options(devices["output"])
        dd_in.value = audio_in if audio_in in devices["input"] else DEFAULT_KEY
        dd_out.value = audio_out if audio_out in devices["output"] else DEFAULT_KEY
        audio_done = on_done
        audio_modal.visible = True
        page.update()

    async def validate_audio(e):
        nonlocal audio_in, audio_out, audio_confirmed, audio_done
        audio_in = "" if dd_in.value in (None, DEFAULT_KEY) else dd_in.value
        audio_out = "" if dd_out.value in (None, DEFAULT_KEY) else dd_out.value
        audio_confirmed = True
        engine.save_prefs({**engine.load_prefs(), "audio": {"input": audio_in, "output": audio_out, "confirmed": True}})
        audio_modal.visible = False
        refresh_audio_summary()
        page.update()
        done, audio_done = audio_done, None
        if done:
            await done()

    def cancel_audio(e):
        nonlocal audio_done
        audio_done = None
        audio_modal.visible = False
        page.update()

    async def apply_audio_change():
        """Nouveau micro/sortie choisis dans les réglages : appliqués tout de suite si Athena tourne."""
        if not assistant: return
        a_in, a_out, was_on = audio_in or None, audio_out or None, mic_on

        def work():
            if was_on:                    # on coupe puis on rallume le micro : le nouvel appareil est pris en compte aussitôt
                assistant.disable_oral()
                time.sleep(0.5)
                assistant.enable_oral(a_in, a_out)
            else:
                assistant.set_audio_devices(a_in, a_out)
        threading.Thread(target=work, daemon=True).start()

    def edit_audio(e):
        open_audio_dialog(apply_audio_change)

    def open_settings(e):
        scrim.visible = panel.visible = True
        page.update()

    def toggle_tokens(e):
        nonlocal show_tokens
        show_tokens = bool(tokens_switch.value)
        if not show_tokens: tokens_box.visible = False
        prefs = engine.load_prefs()
        engine.save_prefs({**prefs, "ui": {**(prefs.get("ui") if isinstance(prefs.get("ui"), dict) else {}), "tokens": show_tokens}})
        page.update()

    def close_settings(e):
        scrim.visible = panel.visible = False
        page.update()

    def on_text_change(e):
        refresh_send()
        page.update()

    async def send_typed_message(e):
        """Entrée dans le champ ou clic sur le bouton d'envoi. Le champ garde (ou récupère) le focus
        pour pouvoir enchaîner les messages sans cliquer à chaque fois."""
        text = text_input.value or ""
        if text.strip() and assistant:
            text_input.value = ""
            refresh_send()
            page.update()
            threading.Thread(target=assistant.process_text, args=(text,), daemon=True).start()
        elif text.strip():                # Athena n'est pas démarrée : on garde le texte et on le dit
            hint.value = "Athena est arrêtée : clique sur « Relancer »."
            hint.color = ACCENT
            page.update()
            await asyncio.sleep(2.5)
            hint.color = TEXT_2
            refresh_hint()
            page.update()
        await text_input.focus()

    mic_btn.on_click = toggle_mic
    send_btn.on_click = send_typed_message
    text_input.on_submit = send_typed_message
    text_input.on_change = on_text_change
    run_btn.on_click = on_run_click
    settings_btn.on_click = open_settings
    def stop_routine(e):
        if assistant: threading.Thread(target=assistant.request_stop, daemon=True).start()     # coupe la voix et arrête la routine

    stop_btn.on_click = stop_routine
    sidebar_btn.on_click = toggle_sidebar
    veille_pill.on_click = veille_click
    new_chat_btn.on_click = new_conversation
    page.on_resize = on_resize
    close_btn.on_click = close_settings
    scrim.on_click = close_settings
    brand.controls.append(tokens_box)      # l'encart se place à droite du mot « Athena », dans l'en-tête
    tokens_switch.on_change = toggle_tokens
    audio_edit_btn.on_click = edit_audio
    audio_ok.on_click = validate_audio
    audio_cancel.on_click = cancel_audio

    # ------------------------------------------------------------------ état initial et assemblage
    refresh_mic()
    refresh_send()
    refresh_hint()
    refresh_welcome()
    set_run_button("off")
    refresh_audio_summary()
    refresh_sidebar()
    apply_layout()

    main_layer = ft.Container(left=0, top=0, right=0, bottom=0, bgcolor=BG,
                              content=ft.Row([sidebar,
                                              ft.Column([header, body, footer], spacing=0, expand=True,
                                                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH)],   # chaque zone prend toute la largeur : la colonne de conversation peut alors se centrer
                                             spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH))
    page.add(ft.Stack([main_layer, scrim, panel, audio_modal], expand=True))
    page.window.prevent_close = True              # la croix passe par shutdown() : arrêt du moteur puis libération de la VRAM
    page.window.on_event = on_window_event
    page.run_task(process_ui_queue)
    start_ia()                                    # Athena démarre toute seule (plus de bouton « Démarrer »)
    threading.Thread(target=engine.prepare_thunderbird_icon, daemon=True).start()   # icône du bouton « Ouvrir Thunderbird » (une seule fois)
    if veille.ACTIVE and veille.CHECK_AT_START:
        threading.Thread(target=veille_startup_check, daemon=True).start()          # compteur de la pastille « News IA » (discret, en arrière-plan)


if __name__ == "__main__":
    import logs
    logs.install()                    # lancement depuis un terminal : l'heure en début de chaque ligne de la console
    ft.run(main, assets_dir=ASSETS_DIR)
