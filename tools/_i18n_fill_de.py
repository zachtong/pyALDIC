"""Fill al_dic_de.ts with German translations.

Conventions (per docs/i18n/glossary.md):
  * German UI uses infinitive verb forms (Speichern, Abbrechen).
  * All nouns capitalised (German grammar rule).
  * Brand / method tokens kept English: pyALDIC, AL-DIC, IC-GN,
    ADMM, FFT, NCC, FEM, Q8, ROI, POI, RMSE, NumPy, MATLAB, CSV,
    NPZ, PNG, PDF.
  * Technical DIC vocabulary: Verschiebung (displacement), Dehnung
    (strain), Netz (mesh), Subset (kept English as it's standard
    in DIC literature), Referenzrahmen (reference frame), Startpunkt
    (starting point).
  * Placeholders (%1, %2, %n, HTML tags <b>) preserved verbatim.
  * German words can be very long — prefer compact forms where
    widget space is tight (e.g. "ROI" over "Interessenbereich").
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_de.ts"

TRANSLATIONS: dict[str, str] = {
    # MainWindow chrome
    "&Settings": "Einstellungen",
    "Language": "Sprache",
    "Language changed": "Sprache geändert",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        "Sprache auf %1 eingestellt. Bitte starten Sie pyALDIC neu, damit alle Elemente die neue Sprache übernehmen.",
    "&File": "Datei",
    "Open Session…": "Sitzung öffnen…",
    "Save Session…": "Sitzung speichern…",
    "Quit": "Beenden",
    "Save Session": "Sitzung speichern",
    "Open Session": "Sitzung öffnen",
    "Save Session Failed": "Speichern der Sitzung fehlgeschlagen",
    "Open Session Failed": "Öffnen der Sitzung fehlgeschlagen",
    "pyALDIC Session": "pyALDIC-Sitzung",
    "All Files": "Alle Dateien",
    "JSON": "JSON",

    # Right sidebar
    "Run DIC Analysis": "DIC-Analyse ausführen",
    "Cancel": "Abbrechen",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "Laufende Analyse abbrechen. Bereits berechnete Frames bleiben erhalten; der Lauf wird als IDLE (nicht DONE) markiert.",
    "Export Results": "Ergebnisse exportieren",
    "Open Strain Window": "Dehnungsfenster öffnen",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "Dehnung in einem separaten Nachbearbeitungsfenster berechnen und visualisieren. Benötigt Verschiebungsergebnisse eines abgeschlossenen Laufs.",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "Platzieren Sie vor dem Ausführen mindestens einen Startpunkt in jeder roten Region (rot = Startpunkt benötigt).",

    # Progress / Field / Visualization
    "PROGRESS": "FORTSCHRITT",
    "Ready": "Bereit",
    "ELAPSED  %1": "VERSTRICHEN  %1",
    "REMAINING  %1": "VERBLEIBEND  %1",
    "%1  —  Frame %2": "%1  —  Frame %2",
    "FIELD": "FELD",
    "Show on deformed frame": "Auf deformiertem Frame anzeigen",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "Wenn aktiviert, werden die Ergebnisse auf dem deformierten (aktuellen) Frame statt auf dem Referenzframe überlagert",
    "VISUALIZATION": "VISUALISIERUNG",
    "Colormap": "Farbkarte",
    "Opacity": "Deckkraft",
    "Overlay opacity (0 = transparent, 100 = opaque)":
        "Deckkraft der Überlagerung (0 = transparent, 100 = deckend)",
    "PHYSICAL UNITS": "PHYSIKALISCHE EINHEITEN",
    "LOG": "PROTOKOLL",
    "Clear": "Leeren",

    # Left sidebar
    "IMAGES": "BILDER",
    "Drop image folder\nor Browse": "Bildordner hier ablegen\noder Durchsuchen",
    "Select Image Folder": "Bildordner auswählen",
    "Natural Sort (1, 2, …, 10)": "Natürliche Sortierung (1, 2, …, 10)",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "Nach eingebetteten Zahlen sortieren: image1, image2, …, image10\nStandard (nicht aktiviert): lexikographisch — ideal für nullgefüllte Namen",
    "WORKFLOW TYPE": "WORKFLOW-TYP",
    "INITIAL GUESS": "STARTSCHÄTZUNG",
    "REGION OF INTEREST": "INTERESSENBEREICH",
    "PARAMETERS": "PARAMETER",
    "ADVANCED": "ERWEITERT",

    # Workflow type panel
    "Tracking Mode": "Tracking-Modus",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "Inkrementell: Jeder Frame wird mit dem vorherigen Referenzframe verglichen.\nGeeignet für große kumulierte Verformungen, erforderlich bei großen Rotationen.\n\nAkkumulativ: Jeder Frame wird mit Frame 1 verglichen.\nNur für kleine, monotone Verformungen genau.",
    "Solver": "Löser",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC: Unabhängiges Subset-Matching (IC-GN). Schnell,\nerhält scharfe lokale Merkmale. Optimal für kleine\nVerformungen oder hochwertige Bilder.\n\nAL-DIC: Augmented Lagrangian mit globaler FEM-\nRegularisierung. Erzwingt Verschiebungskompatibilität\nzwischen Subsets. Optimal für große Verformungen,\nverrauschte Bilder oder hohe Dehnungsgenauigkeit.",
    "Reference Update": "Referenzaktualisierung",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "Zeitpunkt der Referenzaktualisierung beim inkrementellen Tracking.\nJeder Frame: Referenz bei jedem Frame zurücksetzen (kleinste Schrittverschiebung,\nam robustesten für große Verformungen).\nAlle N Frames: alle N Frames zurücksetzen (Balance zwischen Geschwindigkeit und Robustheit).\nBenutzerdefiniert: vom Benutzer festgelegte Liste der Referenz-Frame-Indizes.",
    "Update reference every N frames": "Referenz alle N Frames aktualisieren",
    "Interval": "Intervall",
    "Reference Frames": "Referenzframes",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "Komma-getrennte Frame-Indizes als Referenzframes (0-basiert)",

    # ROI toolbar
    "+ Add": "+ Hinzufügen",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "Region zum Interessenbereich hinzufügen (Polygon / Rechteck / Kreis)",
    "Cut": "Ausschneiden",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "Region aus dem Interessenbereich ausschneiden (Polygon / Rechteck / Kreis)",
    "+ Refine": "+ Verfeinern",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "Zusätzliche Netzverfeinerungszonen mit einem Pinsel malen\n(nur auf Frame 1 — Materialpunkte werden automatisch auf spätere Frames übertragen)",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "Der Verfeinerungspinsel ist nur auf Frame 1 verfügbar. Wechseln Sie zu Frame 1, um Verfeinerungszonen zu malen; sie werden automatisch auf spätere Frames übertragen.",
    "Import": "Importieren",
    "Import mask from image file": "Maske aus Bilddatei importieren",
    "Batch Import": "Stapelimport",
    "Batch import mask files for multiple frames":
        "Maskendateien für mehrere Frames stapelweise importieren",
    "Save": "Speichern",
    "Save current mask to PNG file": "Aktuelle Maske als PNG-Datei speichern",
    "Invert": "Invertieren",
    "Invert the Region of Interest mask": "Maske des Interessenbereichs invertieren",
    "Clear all Region of Interest masks": "Alle Masken des Interessenbereichs leeren",
    "Radius": "Radius",
    "Paint": "Malen",
    "Erase": "Radieren",
    "Clear Brush": "Pinsel leeren",

    # Parameters panel
    "Subset Size": "Subset-Größe",
    "IC-GN subset window size in pixels (odd number)":
        "IC-GN-Subset-Fenstergröße in Pixeln (ungerade Zahl)",
    "Subset Step": "Subset-Schrittweite",
    "Node spacing in pixels (must be power of 2)":
        "Knotenabstand in Pixeln (muss eine Zweierpotenz sein)",
    "Search Range": "Suchbereich",
    "Initial Seed Search": "Anfängliche Seed-Suche",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "Maximale Verschiebung pro Frame, die die FFT-Suche erkennen kann (Pixel).\nDeutlich größer als die erwartete Bewegung zwischen Frames einstellen.\nFür große Rotationen im inkrementellen Modus muss dies abdecken:\n  Radius × sin(Winkel pro Schritt).",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "Anfängliche Halbbreite (Pixel) der Einpunkt-NCC-Suche an jedem Startpunkt.\nErweitert sich bei abgeschnittenem Peak automatisch um den Faktor 2 pro Wiederholung bis zur halben Bildgröße.\nBetrifft nur die Initialisierung der Startpunkte; andere Knoten verwenden F-aware-Propagation (keine knotenweise Suche).",
    "Refine Inner Boundary": "Innere Grenze verfeinern",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "Netz entlang innerer Maskenränder lokal verfeinern\n(Löcher im Interessenbereich). Nützlich für Blasen- oder Porenränder.",
    "Refine Outer Boundary": "Äußere Grenze verfeinern",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "Netz entlang des äußeren Rands des Interessenbereichs lokal verfeinern.",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "Verfeinerungsstärke. Minimale Elementgröße = max(2, subset_step / 2^Stufe). Wird gleichmäßig auf innere, äußere Grenzen UND mit dem Pinsel gemalte Verfeinerungszonen angewendet. Verfügbare Stufen hängen von Subset-Größe und Subset-Schrittweite ab.",
    "Refinement Level": "Verfeinerungsstufe",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "min. Elementgröße = %1 px  (subset_step=%2, Stufe=%3)",

    # Initial guess widget
    "Starting Points": "Startpunkte",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "Platzieren Sie einige Punkte; pyALDIC initialisiert jeden mit einer Einpunkt-NCC und propagiert das Feld entlang der Netz-Nachbarn.\n\nOptimal für:\n• Große Verschiebungen zwischen Frames (> 50 px)\n• Diskontinuierliche Felder (Risse, Scherbänder)\n• Szenarien, in denen FFT falsche Peaks wählt\n\nBeim Zeichnen oder Bearbeiten eines ROI automatisch pro Region platziert.",
    "Place Starting Points": "Startpunkte platzieren",
    "Placing... (click to exit)": "Platzieren… (zum Beenden klicken)",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "Platzierungsmodus auf der Zeichenfläche aktivieren. Linksklick zum Hinzufügen, Rechtsklick zum Entfernen, Esc oder erneuter Klick zum Beenden.",
    "Auto-place": "Auto-Platzieren",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "Leere Regionen mit dem Knoten mit höchstem NCC-Wert füllen. Vorhandene Startpunkte bleiben erhalten.",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "Alle Startpunkte entfernen. Schneller als jeden einzeln per Rechtsklick zu löschen.",
    "%1 / %2 regions ready": "%1 / %2 Regionen bereit",
    "FFT (cross-correlation)": "FFT (Kreuzkorrelation)",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "Normalisierte Kreuzkorrelation auf dem gesamten Gitter. Robust innerhalb des Suchradius; die Suche erweitert sich automatisch bei abgeschnittenen Peaks.\n\nOptimal für:\n• Kleine bis mittlere gleichmäßige Bewegungen\n• Gut texturierte Speckles\n• Keine spezielle Nutzerkonfiguration erforderlich\n\nAufwand wächst mit dem Suchradius, sehr große Verschiebungen werden langsam.",
    "Every": "Alle",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "FFT alle N Frames ausführen. N = 1 bedeutet FFT in jedem Frame (sicherste, langsamste Option). N > 1 verwendet Warmstart zwischen Resets, um die Fehlerausbreitung auf N Frames zu begrenzen.",
    "(N=1 = every frame)": "(N=1 = jeder Frame)",
    "Only when reference frame updates (incremental only)":
        "Nur bei Referenzframe-Aktualisierung (nur inkrementell)",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "FFT bei jedem Wechsel des Referenzframes ausführen; Warmstart innerhalb jedes Segments. Typischer Standard für den inkrementellen Modus.",
    "Previous frame": "Vorheriger Frame",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "Die konvergierte Verschiebung des vorherigen Frames als Startschätzung verwenden. Keine Kreuzkorrelation wird ausgeführt.\n\nOptimal für:\n• Sehr kleine Bewegungen zwischen Frames (wenige Pixel)\n• Schnellste Option bei gleichmäßiger Bewegung\n\nFehler können sich über lange Sequenzen akkumulieren. Bei verrauschten Daten oder größerer Bewegung FFT oder Startpunkte bevorzugen.",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "Laden Sie zuerst Bilder und zeichnen Sie dann einen Interessenbereich auf Frame 1.",
    "<b>Accumulative mode</b> — only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>Akkumulativer Modus</b> — nur Frame 1 benötigt einen Interessenbereich. Alle späteren Frames werden direkt damit verglichen.",
    "<b>Incremental, every frame</b> — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>Inkrementell, jeder Frame</b> — Frame 1 benötigt einen Interessenbereich. Er wird automatisch auf jeden späteren Frame vorwärts übertragen (kein frameweises Zeichnen erforderlich).",
    "<b>Incremental, every %1 frames</b> — draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>Inkrementell, alle %1 Frames</b> — Interessenbereich auf folgenden Frames zeichnen: <b>%2</b> (insgesamt %3 Referenzframes).",
    "<b>Incremental, custom</b> — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>Inkrementell, benutzerdefiniert</b> — keine benutzerdefinierten Referenzframes festgelegt. Frame 1 wird die einzige Referenz sein; fügen Sie weitere Indizes im Feld „Referenzframes“ hinzu.",
    "<b>Incremental, custom</b> — draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>Inkrementell, benutzerdefiniert</b> — Interessenbereich auf folgenden Frames zeichnen: <b>%1</b> (insgesamt %2 Referenzframes).",
    "Draw a Region of Interest on frame 1.": "Zeichnen Sie einen Interessenbereich auf Frame 1.",

    # Export dialog
    "All": "Alle",
    "None": "Keine",
    "OUTPUT FOLDER": "AUSGABEORDNER",
    "Select output folder…": "Ausgabeordner wählen…",
    "Browse…": "Durchsuchen…",
    "Open Folder": "Ordner öffnen",
    "Enable physical units": "Physikalische Einheiten aktivieren",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "Verschiebungswerte mit der Pixelgröße skalieren und physikalische Einheiten auf den Farbleistenbeschriftungen anzeigen. Dehnung ist dimensionslos und wird nicht beeinflusst.",
    "/ pixel": "/ Pixel",
    "Pixel size": "Pixelgröße",
    "fps": "fps",
    "Frame rate": "Framerate",
    "Data": "Daten",
    "Images": "Bilder",
    "Animation": "Animation",
    "Report": "Bericht",
    "FORMAT": "FORMAT",
    "NumPy Archive (.npz)": "NumPy-Archiv (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV (pro Frame)",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ: eine Datei pro Frame (Standard: eine zusammengeführte Datei)",
    "DISPLACEMENT": "VERSCHIEBUNG",
    "Select:": "Auswählen:",
    "STRAIN": "DEHNUNG",
    "Run Compute Strain first.": "Zuerst „Dehnung berechnen“ ausführen.",
    "✓ Parameters file (JSON) always exported": "✓ Parameterdatei (JSON) wird immer exportiert",
    "Export Data": "Daten exportieren",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "Masken des Interessenbereichs stapelweise importieren",
    "Mask Folder:": "Maskenordner:",
    "(none)": "(keine)",
    "Browse...": "Durchsuchen…",
    "Available Masks": "Verfügbare Masken",
    "Auto-Match by Name": "Automatisch nach Name zuordnen",
    "Match mask files to frames by number in filename":
        "Maskendateien anhand der Zahl im Dateinamen Frames zuordnen",
    "Assign Sequential": "Sequentiell zuweisen",
    "Assign masks to frames in order starting from frame 0":
        "Masken den Frames der Reihe nach ab Frame 0 zuweisen",
    "Frame Assignments": "Frame-Zuweisungen",
    "Frame": "Frame",
    "Image": "Bild",
    "Mask": "Maske",
    "Assign Selected ->": "Auswahl zuweisen ->",
    "Pair selected mask(s) with selected frame(s)":
        "Ausgewählte Maske(n) mit ausgewähltem/ausgewählten Frame(s) koppeln",
    "Clear All": "Alle leeren",

    # Canvas area / toolbar
    "Fit": "Anpassen",
    "Fit image to viewport": "Bild an den Ansichtsbereich anpassen",
    "100%": "100%",
    "Zoom to 100% (1:1)": "Auf 100% (1:1) zoomen",
    "Zoom in": "Vergrößern",
    "–": "–",
    "Zoom out": "Verkleinern",
    "Show Grid": "Gitter anzeigen",
    "Show/hide computational mesh grid": "Berechnungsnetz ein-/ausblenden",
    "Show Subset": "Subset anzeigen",
    "Show subset window on hover (requires Grid)":
        "Subset-Fenster beim Überfahren anzeigen (erfordert Gitter)",
    "Placing Starting Points": "Startpunkte werden platziert",

    # Color range
    "Range": "Bereich",
    "Auto": "Auto",
    "Min": "Min",
    "Max": "Max",

    # Field selector
    "Disp U": "Versch. U",
    "Disp V": "Versch. V",

    # Frame navigator
    "Play animation": "Animation abspielen",
    "▶": "▶",
    "Next frame": "Nächster Frame",
    "Playback speed": "Wiedergabegeschwindigkeit",
    "FRAME 0/0": "FRAME 0/0",
    "⏸": "⏸",
    "Pause animation": "Animation pausieren",

    # Image list
    "Add": "Hinzu.",
    "Edit": "Bearb.",
    "Need": "Offen",

    # Mesh appearance
    "Mesh color": "Netzfarbe",
    "Click to choose mesh line color": "Klicken, um die Netzlinienfarbe zu wählen",
    "Line width": "Linienbreite",

    # Physical units widget
    "Use physical units": "Physikalische Einheiten verwenden",
    "Physical size of one image pixel": "Physikalische Größe eines Bildpixels",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)":
        "Aufnahme-Framerate (für das Geschwindigkeitsfeld verwendet)",
    "Disp: px  Velocity: px/fr": "Versch.: px  Geschw.: px/fr",

    # Advanced tuning
    "ADMM Iterations": "ADMM-Iterationen",
    "Number of ADMM alternating minimization cycles for AL-DIC.\n1 = single global pass (fastest), 3 = default,\n5+ = diminishing returns for most cases.":
        "Anzahl der alternierenden ADMM-Minimierungszyklen für AL-DIC.\n1 = einzelner globaler Durchgang (am schnellsten), 3 = Standard,\n5+ = abnehmender Nutzen in den meisten Fällen.",
    "Only affects AL-DIC solver. Ignored by Local DIC.":
        "Betrifft nur den AL-DIC-Löser. Wird von Local DIC ignoriert.",
    "Auto-expand FFT search on clipped peaks":
        "FFT-Suche bei abgeschnittenen Peaks automatisch erweitern",
    "When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).\n\nOnly relevant for the FFT init-guess mode.":
        "Wenn der NCC-Peak den Rand des Suchbereichs erreicht, wird automatisch mit einem größeren Bereich wiederholt (bis zur halben Bildgröße, 6 Versuche mit 2-facher Vergrößerung).\n\nNur relevant für den FFT-Startschätzungsmodus.",

    # Canvas overlay
    "Mode": "Modus",
    "Init": "Start",
    "Accumulative": "Akkumulativ",
    "Incremental": "Inkrementell",
    "Local DIC": "Local DIC",
    "Every Frame": "Jeder Frame",
    "Every N Frames": "Alle N Frames",
    "Custom Frames": "Benutzerdefiniert",
    "ADMM (%1 iter)": "ADMM (%1 Iter.)",
    "FFT every frame": "FFT jeder Frame",
    "FFT every %1 fr": "FFT alle %1 Frames",
    "FFT": "FFT",

    # Log messages
    "Load images first.": "Bitte zuerst Bilder laden.",
    "Building pipeline configuration...": "Pipeline-Konfiguration wird erstellt…",
    "Loading images...": "Bilder werden geladen…",

    # Strain window
    "Strain Post-Processing": "Dehnungs-Nachbearbeitung",
    "Compute Strain": "Dehnung berechnen",
    "Starting…": "Wird gestartet…",
    "Complete": "Abgeschlossen",
    "⚠ Params changed -- click Compute Strain": "⚠ Parameter geändert — „Dehnung berechnen“ klicken",
    "Unit: px/frame": "Einheit: px/Frame",

    # Strain field selector
    "STRAIN PARAMETERS": "DEHNUNGSPARAMETER",

    # Strain param panel
    "Method": "Methode",
    "Plane fitting": "Ebenenanpassung",
    "FEM nodal": "FEM-Knoten",
    "VSG size": "VSG-Größe",
    "Strain field smoothing": "Dehnungsfeldglättung",
    "Strain type": "Dehnungstyp",
    "Infinitesimal": "Infinitesimal",
    "Eulerian": "Euler",
    "Green-Lagrangian": "Green-Lagrange",
    "Gaussian smoothing of the strain field after computation.\nσ is the Gaussian kernel width; 'step' = DIC node spacing.\n  Light  (0.5 × step):  subtle, preserves fine features.\n  Medium (1 × step):    balanced, recommended for noisy data.\n  Strong (2 × step) ⚠:  aggressive, may blur real gradients.":
        "Gauß-Glättung des Dehnungsfelds nach der Berechnung.\nσ ist die Breite des Gauß-Kerns; „step“ = DIC-Knotenabstand.\n  Leicht   (0,5 × step): dezent, feine Merkmale bleiben erhalten.\n  Mittel   (1 × step):   ausgewogen, empfohlen für verrauschte Daten.\n  Stark    (2 × step) ⚠: aggressiv, kann echte Gradienten verwischen.",

    # Strain window export tooltip
    "Export displacement and strain results to NPZ / MAT / CSV / PNG":
        "Verschiebungs- und Dehnungsergebnisse als NPZ / MAT / CSV / PNG exportieren",
}


def main() -> None:
    content = TS.read_text(encoding="utf-8")
    filled = 0
    miss: list[str] = []

    def _repl(m: re.Match) -> str:
        nonlocal filled
        source = m.group(1)
        key = (source
               .replace("&amp;", "&").replace("&lt;", "<")
               .replace("&gt;", ">").replace("&quot;", '"')
               .replace("&apos;", "'"))
        target = TRANSLATIONS.get(key)
        if target is None:
            miss.append(key)
            return m.group(0)
        escaped = (target
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        filled += 1
        return (f"<source>{source}</source>\n"
                f"        <translation>{escaped}</translation>")

    new_content = re.sub(
        r"<source>(.*?)</source>\s*"
        r"<translation(?: type=\"unfinished\")?>([^<]*)</translation>",
        _repl,
        content,
        flags=re.DOTALL,
    )
    TS.write_text(new_content, encoding="utf-8")
    print(f"[de] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
