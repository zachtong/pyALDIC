"""Fill al_dic_fr.ts with French translations.

Conventions (per docs/i18n/glossary.md):
  * French UI uses infinitive verb forms (Enregistrer, Annuler).
  * Non-breaking space (U+00A0) before ':', ';', '?', '!' per
    French typography; colons/punctuation inside GUI strings use
    a regular space to stay compatible with narrow label widths.
  * Brand tokens kept English: pyALDIC, AL-DIC, IC-GN, ADMM, FFT,
    NCC, FEM, Q8, ROI, POI, RMSE, NumPy, MATLAB, CSV, NPZ, PNG, PDF.
  * Technical DIC vocabulary: Déplacement, Déformation, Maillage,
    Imagette / Sous-ensemble, Frame, Point de départ.
  * Placeholders (%1, %2, %n, HTML tags <b>) preserved verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_fr.ts"

# Non-breaking space before French punctuation
NB = "\u00a0"

TRANSLATIONS: dict[str, str] = {
    # MainWindow chrome
    "&Settings": "Paramètres",
    "Language": "Langue",
    "Language changed": "Langue modifiée",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        f"Langue définie sur %1. Veuillez redémarrer pyALDIC pour que tous les éléments prennent en compte la nouvelle langue.",
    "&File": "Fichier",
    "Open Session…": "Ouvrir une session…",
    "Save Session…": "Enregistrer la session…",
    "Quit": "Quitter",
    "Save Session": "Enregistrer la session",
    "Open Session": "Ouvrir une session",
    "Save Session Failed": "Échec de l'enregistrement de la session",
    "Open Session Failed": "Échec de l'ouverture de la session",
    "pyALDIC Session": "Session pyALDIC",
    "All Files": "Tous les fichiers",
    "JSON": "JSON",

    # Right sidebar
    "Run DIC Analysis": "Lancer l'analyse DIC",
    "Cancel": "Annuler",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "Annuler l'analyse en cours. Les images déjà calculées sont conservées ; l'exécution passe à l'état IDLE (et non DONE).",
    "Export Results": "Exporter les résultats",
    "Open Strain Window": "Ouvrir la fenêtre de déformation",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "Calculer et visualiser la déformation dans une fenêtre de post-traitement séparée. Nécessite des résultats de déplacement d'une exécution terminée.",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "Placez au moins un point de départ dans chaque région rouge avant de lancer l'exécution (rouge = point de départ requis).",

    # Progress / Field / Visualization
    "PROGRESS": "PROGRESSION",
    "Ready": "Prêt",
    "ELAPSED  %1": "ÉCOULÉ  %1",
    "REMAINING  %1": "RESTANT  %1",
    "%1  —  Frame %2": "%1  —  Image %2",
    "FIELD": "CHAMP",
    "Show on deformed frame": "Afficher sur l'image déformée",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "Lorsque cette option est activée, les résultats sont superposés sur l'image déformée (actuelle) au lieu de l'image de référence",
    "VISUALIZATION": "VISUALISATION",
    "Colormap": "Palette de couleurs",
    "Opacity": "Opacité",
    "Overlay opacity (0 = transparent, 100 = opaque)":
        "Opacité de la superposition (0 = transparent, 100 = opaque)",
    "PHYSICAL UNITS": "UNITÉS PHYSIQUES",
    "LOG": "JOURNAL",
    "Clear": "Effacer",

    # Left sidebar
    "IMAGES": "IMAGES",
    "Drop image folder\nor Browse": "Déposez un dossier d'images\nou Parcourir",
    "Select Image Folder": "Sélectionner le dossier d'images",
    "Natural Sort (1, 2, …, 10)": "Tri naturel (1, 2, …, 10)",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "Tri par numéros intégrés : image1, image2, …, image10\nPar défaut (non coché) : lexicographique — idéal pour les noms avec zéros de remplissage",
    "WORKFLOW TYPE": "TYPE DE FLUX",
    "INITIAL GUESS": "ESTIMATION INITIALE",
    "REGION OF INTEREST": "RÉGION D'INTÉRÊT",
    "PARAMETERS": "PARAMÈTRES",
    "ADVANCED": "AVANCÉ",

    # Workflow type panel
    "Tracking Mode": "Mode de suivi",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "Incrémental : chaque image est comparée à l'image de référence précédente.\nAdapté aux grandes déformations cumulées, requis pour les grandes rotations.\n\nCumulatif : chaque image est comparée à l'image 1.\nPrécis uniquement pour les petites déformations monotones.",
    "Solver": "Solveur",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC : Appariement d'imagettes indépendant (IC-GN). Rapide,\npréserve les détails locaux. Idéal pour les petites\ndéformations ou les images de haute qualité.\n\nAL-DIC : Lagrangien augmenté avec régularisation\nFEM globale. Impose la compatibilité des déplacements\nentre imagettes. Idéal pour les grandes déformations, les images\nbruitées ou lorsque la précision de la déformation est importante.",
    "Reference Update": "Mise à jour de la référence",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "Quand l'image de référence est rafraîchie lors du suivi incrémental.\nChaque image : réinitialiser la référence à chaque image (plus petit déplacement par étape,\nplus robuste pour les grandes déformations).\nToutes les N images : réinitialiser toutes les N images (compromis vitesse/robustesse).\nImages personnalisées : liste d'indices définis par l'utilisateur.",
    "Update reference every N frames": "Mettre à jour la référence toutes les N images",
    "Interval": "Intervalle",
    "Reference Frames": "Images de référence",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "Indices d'images séparés par des virgules pour les images de référence (base 0)",

    # ROI toolbar
    "+ Add": "+ Ajouter",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "Ajouter une région à la région d'intérêt (Polygone / Rectangle / Cercle)",
    "Cut": "Découper",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "Découper une région de la région d'intérêt (Polygone / Rectangle / Cercle)",
    "+ Refine": "+ Raffiner",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "Peindre des zones de raffinage de maillage supplémentaires au pinceau\n(uniquement sur l'image 1 — les points matériels sont automatiquement reportés sur les images suivantes)",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "Le pinceau de raffinage n'est disponible que sur l'image 1. Passez à l'image 1 pour peindre les zones de raffinage ; elles sont automatiquement reportées sur les images suivantes.",
    "Import": "Importer",
    "Import mask from image file": "Importer le masque depuis un fichier image",
    "Batch Import": "Import par lot",
    "Batch import mask files for multiple frames":
        "Importer par lot des fichiers de masques pour plusieurs images",
    "Save": "Enregistrer",
    "Save current mask to PNG file": "Enregistrer le masque actuel en PNG",
    "Invert": "Inverser",
    "Invert the Region of Interest mask": "Inverser le masque de la région d'intérêt",
    "Clear all Region of Interest masks": "Effacer tous les masques de région d'intérêt",
    "Radius": "Rayon",
    "Paint": "Peindre",
    "Erase": "Effacer",
    "Clear Brush": "Effacer le pinceau",

    # Parameters panel
    "Subset Size": "Taille d'imagette",
    "IC-GN subset window size in pixels (odd number)":
        "Taille de la fenêtre d'imagette IC-GN en pixels (nombre impair)",
    "Subset Step": "Pas d'imagette",
    "Node spacing in pixels (must be power of 2)":
        "Espacement des nœuds en pixels (doit être une puissance de 2)",
    "Search Range": "Plage de recherche",
    "Initial Seed Search": "Recherche initiale du germe",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "Déplacement maximal par image détectable par la recherche FFT (pixels).\nDéfinissez une valeur nettement supérieure au mouvement inter-image attendu.\nPour les grandes rotations en mode incrémental, cela doit couvrir :\n  rayon × sin(angle par étape).",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "Demi-largeur initiale (pixels) de la recherche NCC mono-point à chaque point de départ.\nS'étend automatiquement d'un facteur 2 par tentative si le pic est tronqué, jusqu'à la moitié de la taille de l'image.\nN'affecte que l'initialisation des points de départ ; les autres nœuds utilisent la propagation F-aware (pas de recherche par nœud).",
    "Refine Inner Boundary": "Raffiner la limite interne",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "Raffiner localement le maillage le long des limites internes du masque\n(trous à l'intérieur de la région d'intérêt). Utile pour les bords de bulles / vides.",
    "Refine Outer Boundary": "Raffiner la limite externe",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "Raffiner localement le maillage le long de la limite externe de la région d'intérêt.",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "Intensité du raffinage. Taille minimale d'élément = max(2, subset_step / 2^niveau). S'applique uniformément aux limites internes, externes ET aux zones peintes au pinceau. Les niveaux disponibles dépendent de la taille et du pas d'imagette.",
    "Refinement Level": "Niveau de raffinage",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "taille min. d'élément = %1 px  (subset_step=%2, niveau=%3)",

    # Initial guess widget
    "Starting Points": "Points de départ",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "Placez quelques points ; pyALDIC initialise chacun avec une NCC mono-point et propage le champ le long des voisins du maillage.\n\nIdéal pour :\n• Grands déplacements inter-images (> 50 px)\n• Champs discontinus (fissures, bandes de cisaillement)\n• Scénarios où la FFT choisit de mauvais pics\n\nPlacés automatiquement par région lorsque vous dessinez ou modifiez une ROI.",
    "Place Starting Points": "Placer les points de départ",
    "Placing... (click to exit)": "Placement… (cliquez pour sortir)",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "Entrer en mode placement sur le canevas. Clic gauche pour ajouter, clic droit pour supprimer, Échap ou nouveau clic pour sortir.",
    "Auto-place": "Placement automatique",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "Remplit les régions vides avec le nœud ayant la meilleure NCC. Les points de départ existants sont conservés.",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "Supprimer tous les points de départ. Plus rapide que de cliquer droit sur chacun.",
    "%1 / %2 regions ready": "%1 / %2 régions prêtes",
    "FFT (cross-correlation)": "FFT (corrélation croisée)",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "Corrélation croisée normalisée sur la grille complète. Robuste dans le rayon de recherche ; la recherche s'étend automatiquement lorsque les pics sont tronqués.\n\nIdéal pour :\n• Mouvements lisses petits à modérés\n• Speckle bien texturé\n• Aucune configuration spéciale requise\n\nLe coût augmente avec le rayon de recherche, les très grands déplacements deviennent donc lents.",
    "Every": "Toutes les",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "Exécute la FFT toutes les N images. N = 1 signifie FFT à chaque image (le plus sûr, le plus lent). N > 1 utilise un démarrage à chaud entre les réinitialisations pour limiter la propagation d'erreurs à N images.",
    "(N=1 = every frame)": "(N=1 = chaque image)",
    "Only when reference frame updates (incremental only)":
        "Uniquement à la mise à jour de la référence (incrémental seulement)",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "Exécute la FFT à chaque changement d'image de référence ; démarrage à chaud dans chaque segment. Valeur par défaut typique pour le mode incrémental.",
    "Previous frame": "Image précédente",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "Utilise le déplacement convergé de l'image précédente comme estimation initiale. Aucune corrélation croisée n'est exécutée.\n\nIdéal pour :\n• Très petits mouvements inter-images (quelques pixels)\n• Option la plus rapide lorsque le mouvement est lisse\n\nLes erreurs peuvent s'accumuler sur les séquences longues. Préférez FFT ou les points de départ sur des données bruitées ou lorsque le mouvement est plus important.",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "Chargez d'abord des images, puis dessinez une région d'intérêt sur l'image 1.",
    "<b>Accumulative mode</b> — only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>Mode cumulatif</b> — seule l'image 1 a besoin d'une région d'intérêt. Toutes les images suivantes sont comparées directement à celle-ci.",
    "<b>Incremental, every frame</b> — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>Incrémental, chaque image</b> — l'image 1 a besoin d'une région d'intérêt. Elle est automatiquement propagée à chaque image suivante (pas de dessin par image requis).",
    "<b>Incremental, every %1 frames</b> — draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>Incrémental, toutes les %1 images</b> — dessinez une région d'intérêt sur les images : <b>%2</b> (%3 images de référence au total).",
    "<b>Incremental, custom</b> — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>Incrémental, personnalisé</b> — aucune image de référence personnalisée définie. L'image 1 sera la seule référence ; ajoutez d'autres indices dans le champ Images de référence.",
    "<b>Incremental, custom</b> — draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>Incrémental, personnalisé</b> — dessinez une région d'intérêt sur les images : <b>%1</b> (%2 images de référence au total).",
    "Draw a Region of Interest on frame 1.": "Dessinez une région d'intérêt sur l'image 1.",

    # Export dialog
    "All": "Tout",
    "None": "Aucun",
    "OUTPUT FOLDER": "DOSSIER DE SORTIE",
    "Select output folder…": "Sélectionner le dossier de sortie…",
    "Browse…": "Parcourir…",
    "Open Folder": "Ouvrir le dossier",
    "Enable physical units": "Activer les unités physiques",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "Mettre à l'échelle les valeurs de déplacement par la taille du pixel et afficher les unités physiques sur les étiquettes de la barre de couleurs. La déformation est sans dimension et n'est pas affectée.",
    "/ pixel": "/ pixel",
    "Pixel size": "Taille du pixel",
    "fps": "fps",
    "Frame rate": "Cadence",
    "Data": "Données",
    "Images": "Images",
    "Animation": "Animation",
    "Report": "Rapport",
    "FORMAT": "FORMAT",
    "NumPy Archive (.npz)": "Archive NumPy (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV (par image)",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ : un fichier par image (par défaut : un seul fichier fusionné)",
    "DISPLACEMENT": "DÉPLACEMENT",
    "Select:": "Sélectionner :",
    "STRAIN": "DÉFORMATION",
    "Run Compute Strain first.": "Exécutez d'abord « Calculer la déformation ».",
    "✓ Parameters file (JSON) always exported": "✓ Le fichier de paramètres (JSON) est toujours exporté",
    "Export Data": "Exporter les données",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "Import par lot des masques de région d'intérêt",
    "Mask Folder:": "Dossier de masques :",
    "(none)": "(aucun)",
    "Browse...": "Parcourir…",
    "Available Masks": "Masques disponibles",
    "Auto-Match by Name": "Correspondance auto par nom",
    "Match mask files to frames by number in filename":
        "Associer les fichiers de masque aux images d'après le numéro du nom de fichier",
    "Assign Sequential": "Attribuer séquentiellement",
    "Assign masks to frames in order starting from frame 0":
        "Attribuer les masques aux images dans l'ordre à partir de l'image 0",
    "Frame Assignments": "Attributions d'images",
    "Frame": "Image",
    "Image": "Image",
    "Mask": "Masque",
    "Assign Selected ->": "Attribuer la sélection ->",
    "Pair selected mask(s) with selected frame(s)":
        "Associer les masques sélectionnés aux images sélectionnées",
    "Clear All": "Tout effacer",

    # Canvas area / toolbar
    "Fit": "Ajuster",
    "Fit image to viewport": "Ajuster l'image à la vue",
    "100%": "100%",
    "Zoom to 100% (1:1)": "Zoomer à 100% (1:1)",
    "Zoom in": "Zoom avant",
    "–": "–",
    "Zoom out": "Zoom arrière",
    "Show Grid": "Afficher la grille",
    "Show/hide computational mesh grid": "Afficher/masquer la grille du maillage",
    "Show Subset": "Afficher l'imagette",
    "Show subset window on hover (requires Grid)":
        "Afficher la fenêtre d'imagette au survol (nécessite la grille)",
    "Placing Starting Points": "Placement des points de départ",

    # Color range
    "Range": "Plage",
    "Auto": "Auto",
    "Min": "Min",
    "Max": "Max",

    # Field selector
    "Disp U": "Dépl. U",
    "Disp V": "Dépl. V",

    # Frame navigator
    "Play animation": "Lire l'animation",
    "▶": "▶",
    "Next frame": "Image suivante",
    "Playback speed": "Vitesse de lecture",
    "FRAME 0/0": "IMAGE 0/0",
    "⏸": "⏸",
    "Pause animation": "Pause",

    # Image list
    "Add": "Ajouter",
    "Edit": "Modifier",
    "Need": "Requis",

    # Mesh appearance
    "Mesh color": "Couleur du maillage",
    "Click to choose mesh line color": "Cliquer pour choisir la couleur des lignes du maillage",
    "Line width": "Épaisseur",

    # Physical units widget
    "Use physical units": "Utiliser les unités physiques",
    "Physical size of one image pixel": "Taille physique d'un pixel de l'image",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)":
        "Cadence d'acquisition (utilisée pour le champ de vitesse)",
    "Disp: px  Velocity: px/fr": "Dépl. : px  Vitesse : px/im",

    # Advanced tuning
    "ADMM Iterations": "Itérations ADMM",
    "Number of ADMM alternating minimization cycles for AL-DIC.\n1 = single global pass (fastest), 3 = default,\n5+ = diminishing returns for most cases.":
        "Nombre de cycles de minimisation alternée ADMM pour AL-DIC.\n1 = une seule passe globale (le plus rapide), 3 = par défaut,\n5+ = rendement décroissant dans la plupart des cas.",
    "Only affects AL-DIC solver. Ignored by Local DIC.":
        "N'affecte que le solveur AL-DIC. Ignoré par Local DIC.",
    "Auto-expand FFT search on clipped peaks":
        "Étendre automatiquement la recherche FFT lors de pics tronqués",
    "When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).\n\nOnly relevant for the FFT init-guess mode.":
        "Lorsque le pic NCC atteint le bord de la zone de recherche, réessaie automatiquement avec une zone plus large (jusqu'à la moitié de l'image, 6 tentatives avec une croissance de 2×).\n\nUniquement pertinent pour le mode d'estimation initiale FFT.",

    # Canvas overlay
    "Mode": "Mode",
    "Init": "Initial",
    "Accumulative": "Cumulatif",
    "Incremental": "Incrémental",
    "Local DIC": "Local DIC",
    "Every Frame": "Chaque image",
    "Every N Frames": "Toutes les N images",
    "Custom Frames": "Images personnalisées",
    "ADMM (%1 iter)": "ADMM (%1 itér.)",
    "FFT every frame": "FFT chaque image",
    "FFT every %1 fr": "FFT toutes les %1 images",
    "FFT": "FFT",

    # Log messages
    "Load images first.": "Veuillez d'abord charger des images.",
    "Building pipeline configuration...": "Construction de la configuration du pipeline…",
    "Loading images...": "Chargement des images…",

    # Strain window
    "Strain Post-Processing": "Post-traitement de la déformation",
    "Compute Strain": "Calculer la déformation",
    "Starting…": "Démarrage…",
    "Complete": "Terminé",
    "⚠ Params changed -- click Compute Strain": "⚠ Paramètres modifiés — cliquez sur « Calculer la déformation »",
    "Unit: px/frame": "Unité : px/image",

    # Strain field selector
    "STRAIN PARAMETERS": "PARAMÈTRES DE DÉFORMATION",

    # Strain param panel
    "Method": "Méthode",
    "Plane fitting": "Ajustement de plan",
    "FEM nodal": "FEM nodal",
    "VSG size": "Taille VSG",
    "Strain field smoothing": "Lissage du champ de déformation",
    "Strain type": "Type de déformation",
    "Infinitesimal": "Infinitésimal",
    "Eulerian": "Eulérien",
    "Green-Lagrangian": "Green-Lagrange",
    "Gaussian smoothing of the strain field after computation.\nσ is the Gaussian kernel width; 'step' = DIC node spacing.\n  Light  (0.5 × step):  subtle, preserves fine features.\n  Medium (1 × step):    balanced, recommended for noisy data.\n  Strong (2 × step) ⚠:  aggressive, may blur real gradients.":
        "Lissage gaussien du champ de déformation après calcul.\nσ est la largeur du noyau gaussien ; « step » = espacement des nœuds DIC.\n  Léger   (0,5 × step) : subtil, préserve les détails fins.\n  Moyen   (1 × step) :   équilibré, recommandé pour des données bruitées.\n  Fort    (2 × step) ⚠ : agressif, peut flouter les vrais gradients.",

    # Strain window export tooltip
    "Export displacement and strain results to NPZ / MAT / CSV / PNG":
        "Exporter les résultats de déplacement et de déformation en NPZ / MAT / CSV / PNG",
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
    print(f"[fr] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
