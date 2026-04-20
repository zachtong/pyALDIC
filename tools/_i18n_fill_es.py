"""Fill al_dic_es.ts with Spanish translations.

Conventions (per docs/i18n/glossary.md):
  * Spanish UI uses imperative / infinitive forms depending on
    context. Button labels prefer short imperatives (Guardar,
    Cancelar, Abrir).
  * Latin-American-neutral vocabulary (fotograma, archivo,
    desplazamiento, deformación).
  * Brand tokens kept English: pyALDIC, AL-DIC, IC-GN, ADMM, FFT,
    NCC, FEM, Q8, ROI, POI, RMSE, NumPy, MATLAB, CSV, NPZ, PNG, PDF.
  * Technical DIC vocabulary: Desplazamiento, Deformación, Malla,
    Subconjunto, Fotograma, Punto de inicio.
  * Placeholders (%1, %2, %n, HTML tags <b>) preserved verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_es.ts"

TRANSLATIONS: dict[str, str] = {
    # MainWindow chrome
    "&Settings": "Configuración",
    "Language": "Idioma",
    "Language changed": "Idioma cambiado",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        "Idioma establecido en %1. Reinicie pyALDIC para que todos los elementos adopten el nuevo idioma.",
    "&File": "Archivo",
    "Open Session…": "Abrir sesión…",
    "Save Session…": "Guardar sesión…",
    "Quit": "Salir",
    "Save Session": "Guardar sesión",
    "Open Session": "Abrir sesión",
    "Save Session Failed": "Error al guardar la sesión",
    "Open Session Failed": "Error al abrir la sesión",
    "pyALDIC Session": "Sesión de pyALDIC",
    "All Files": "Todos los archivos",
    "JSON": "JSON",

    # Right sidebar
    "Run DIC Analysis": "Ejecutar análisis DIC",
    "Cancel": "Cancelar",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "Cancelar el análisis actual. Los fotogramas ya calculados se conservan; la ejecución se marca como IDLE (no DONE).",
    "Export Results": "Exportar resultados",
    "Open Strain Window": "Abrir ventana de deformación",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "Calcular y visualizar la deformación en una ventana de post-procesado separada. Requiere resultados de desplazamiento de una ejecución completada.",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "Coloque al menos un punto de inicio en cada región roja antes de ejecutar (rojo = requiere punto de inicio).",

    # Progress / Field / Visualization
    "PROGRESS": "PROGRESO",
    "Ready": "Listo",
    "ELAPSED  %1": "TRANSCURRIDO  %1",
    "REMAINING  %1": "RESTANTE  %1",
    "%1  —  Frame %2": "%1  —  Fotograma %2",
    "FIELD": "CAMPO",
    "Show on deformed frame": "Mostrar en fotograma deformado",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "Si está activado, los resultados se superponen sobre el fotograma deformado (actual) en lugar del fotograma de referencia",
    "VISUALIZATION": "VISUALIZACIÓN",
    "Colormap": "Paleta de colores",
    "Opacity": "Opacidad",
    "Overlay opacity (0 = transparent, 100 = opaque)":
        "Opacidad de la superposición (0 = transparente, 100 = opaco)",
    "PHYSICAL UNITS": "UNIDADES FÍSICAS",
    "LOG": "REGISTRO",
    "Clear": "Limpiar",

    # Left sidebar
    "IMAGES": "IMÁGENES",
    "Drop image folder\nor Browse": "Arrastre la carpeta de imágenes\no Examinar",
    "Select Image Folder": "Seleccionar carpeta de imágenes",
    "Natural Sort (1, 2, …, 10)": "Orden natural (1, 2, …, 10)",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "Ordenar por números incrustados: image1, image2, …, image10\nPredeterminado (desmarcado): lexicográfico — ideal para nombres con ceros a la izquierda",
    "WORKFLOW TYPE": "TIPO DE FLUJO",
    "INITIAL GUESS": "ESTIMACIÓN INICIAL",
    "REGION OF INTEREST": "REGIÓN DE INTERÉS",
    "PARAMETERS": "PARÁMETROS",
    "ADVANCED": "AVANZADO",

    # Workflow type panel
    "Tracking Mode": "Modo de seguimiento",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "Incremental: cada fotograma se compara con el fotograma de referencia anterior.\nAdecuado para grandes deformaciones acumuladas; obligatorio en grandes rotaciones.\n\nAcumulativo: cada fotograma se compara con el fotograma 1.\nPreciso solo para deformaciones pequeñas y monótonas.",
    "Solver": "Solucionador",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC: Coincidencia de subconjuntos independiente (IC-GN). Rápido,\nconserva detalles locales nítidos. Ideal para pequeñas\ndeformaciones o imágenes de alta calidad.\n\nAL-DIC: Lagrangiano aumentado con regularización\nFEM global. Impone compatibilidad de desplazamientos\nentre subconjuntos. Ideal para grandes deformaciones, imágenes\ncon ruido o cuando la precisión de la deformación es importante.",
    "Reference Update": "Actualización de la referencia",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "Cuándo se actualiza el fotograma de referencia durante el seguimiento incremental.\nCada fotograma: reiniciar la referencia en cada fotograma (menor desplazamiento por paso,\nmás robusto para grandes deformaciones).\nCada N fotogramas: reiniciar cada N fotogramas (equilibrio entre velocidad y robustez).\nFotogramas personalizados: lista de índices definida por el usuario.",
    "Update reference every N frames": "Actualizar la referencia cada N fotogramas",
    "Interval": "Intervalo",
    "Reference Frames": "Fotogramas de referencia",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "Índices de fotograma separados por comas para usar como fotogramas de referencia (base 0)",

    # ROI toolbar
    "+ Add": "+ Añadir",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "Añadir región a la región de interés (Polígono / Rectángulo / Círculo)",
    "Cut": "Recortar",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "Recortar región de la región de interés (Polígono / Rectángulo / Círculo)",
    "+ Refine": "+ Refinar",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "Pintar zonas adicionales de refinamiento de malla con un pincel\n(solo en el fotograma 1 — los puntos materiales se propagan automáticamente a los fotogramas posteriores)",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "El pincel de refinamiento solo está disponible en el fotograma 1. Cambie al fotograma 1 para pintar zonas de refinamiento; se propagan automáticamente a los fotogramas posteriores.",
    "Import": "Importar",
    "Import mask from image file": "Importar máscara desde archivo de imagen",
    "Batch Import": "Importación por lotes",
    "Batch import mask files for multiple frames":
        "Importar por lotes archivos de máscara para varios fotogramas",
    "Save": "Guardar",
    "Save current mask to PNG file": "Guardar la máscara actual en archivo PNG",
    "Invert": "Invertir",
    "Invert the Region of Interest mask": "Invertir la máscara de la región de interés",
    "Clear all Region of Interest masks": "Limpiar todas las máscaras de región de interés",
    "Radius": "Radio",
    "Paint": "Pintar",
    "Erase": "Borrar",
    "Clear Brush": "Limpiar pincel",

    # Parameters panel
    "Subset Size": "Tamaño del subconjunto",
    "IC-GN subset window size in pixels (odd number)":
        "Tamaño de la ventana del subconjunto IC-GN en píxeles (número impar)",
    "Subset Step": "Paso del subconjunto",
    "Node spacing in pixels (must be power of 2)":
        "Espaciado de nodos en píxeles (debe ser potencia de 2)",
    "Search Range": "Rango de búsqueda",
    "Initial Seed Search": "Búsqueda inicial de semilla",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "Desplazamiento máximo por fotograma que puede detectar la búsqueda FFT (píxeles).\nConfigúrelo claramente mayor que el movimiento esperado entre fotogramas.\nPara grandes rotaciones en modo incremental, debe cubrir:\n  radio × sin(ángulo por paso).",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "Semianchura inicial (píxeles) de la búsqueda NCC puntual en cada punto de inicio.\nSe expande automáticamente 2× por reintento si el pico se recorta, hasta la mitad del tamaño de la imagen.\nSolo afecta a la inicialización de los puntos de inicio; los demás nodos usan propagación F-aware (sin búsqueda por nodo).",
    "Refine Inner Boundary": "Refinar borde interior",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "Refinar localmente la malla a lo largo de los bordes internos de la máscara\n(agujeros dentro de la región de interés). Útil para bordes de burbujas o huecos.",
    "Refine Outer Boundary": "Refinar borde exterior",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "Refinar localmente la malla a lo largo del borde exterior de la región de interés.",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "Intensidad del refinamiento. Tamaño mínimo de elemento = max(2, subset_step / 2^nivel). Se aplica uniformemente a bordes interiores, exteriores Y zonas pintadas con el pincel. Los niveles disponibles dependen del tamaño y el paso del subconjunto.",
    "Refinement Level": "Nivel de refinamiento",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "tamaño mín. de elemento = %1 px  (subset_step=%2, nivel=%3)",

    # Initial guess widget
    "Starting Points": "Puntos de inicio",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "Coloque algunos puntos; pyALDIC inicializa cada uno con una NCC puntual y propaga el campo a lo largo de los vecinos de la malla.\n\nIdeal para:\n• Grandes desplazamientos entre fotogramas (> 50 px)\n• Campos discontinuos (grietas, bandas de cortante)\n• Escenarios donde la FFT elige picos incorrectos\n\nSe colocan automáticamente por región al dibujar o editar una ROI.",
    "Place Starting Points": "Colocar puntos de inicio",
    "Placing... (click to exit)": "Colocando… (clic para salir)",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "Entrar en modo de colocación en el lienzo. Clic izquierdo para añadir, clic derecho para eliminar, Esc o nuevo clic para salir.",
    "Auto-place": "Colocación automática",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "Rellenar las regiones vacías con el nodo de mayor NCC en cada una. Se conservan los puntos de inicio existentes.",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "Eliminar todos los puntos de inicio. Más rápido que hacer clic derecho en cada uno.",
    "%1 / %2 regions ready": "%1 / %2 regiones listas",
    "FFT (cross-correlation)": "FFT (correlación cruzada)",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "Correlación cruzada normalizada en toda la cuadrícula. Robusta dentro del radio de búsqueda; la búsqueda se expande automáticamente cuando los picos se recortan.\n\nIdeal para:\n• Movimientos suaves pequeños o moderados\n• Moteado bien texturado\n• No se requiere configuración especial del usuario\n\nEl coste crece con el radio de búsqueda, por lo que desplazamientos muy grandes se vuelven lentos.",
    "Every": "Cada",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "Ejecutar FFT cada N fotogramas. N = 1 significa FFT en cada fotograma (más seguro, más lento). N > 1 usa arranque en caliente entre reinicios para limitar la propagación de errores a N fotogramas.",
    "(N=1 = every frame)": "(N=1 = cada fotograma)",
    "Only when reference frame updates (incremental only)":
        "Solo cuando se actualiza el fotograma de referencia (solo incremental)",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "Ejecutar FFT siempre que cambie el fotograma de referencia; arranque en caliente dentro de cada segmento. Valor predeterminado típico para el modo incremental.",
    "Previous frame": "Fotograma anterior",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "Usar el desplazamiento convergido del fotograma anterior como estimación inicial. No se ejecuta correlación cruzada.\n\nIdeal para:\n• Movimientos entre fotogramas muy pequeños (unos pocos píxeles)\n• La opción más rápida cuando el movimiento es suave\n\nLos errores pueden acumularse en secuencias largas. Prefiera FFT o puntos de inicio con datos ruidosos o cuando el movimiento sea mayor.",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "Cargue primero las imágenes y luego dibuje una región de interés en el fotograma 1.",
    "<b>Accumulative mode</b> — only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>Modo acumulativo</b> — solo el fotograma 1 requiere una región de interés. Todos los fotogramas posteriores se comparan directamente con ella.",
    "<b>Incremental, every frame</b> — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>Incremental, cada fotograma</b> — el fotograma 1 requiere una región de interés. Se propaga automáticamente hacia cada fotograma posterior (no es necesario dibujar por fotograma).",
    "<b>Incremental, every %1 frames</b> — draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>Incremental, cada %1 fotogramas</b> — dibuje una región de interés en los fotogramas: <b>%2</b> (%3 fotogramas de referencia en total).",
    "<b>Incremental, custom</b> — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>Incremental, personalizado</b> — no hay fotogramas de referencia personalizados definidos. El fotograma 1 será la única referencia; añada más índices en el campo «Fotogramas de referencia».",
    "<b>Incremental, custom</b> — draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>Incremental, personalizado</b> — dibuje una región de interés en los fotogramas: <b>%1</b> (%2 fotogramas de referencia en total).",
    "Draw a Region of Interest on frame 1.": "Dibuje una región de interés en el fotograma 1.",

    # Export dialog
    "All": "Todos",
    "None": "Ninguno",
    "OUTPUT FOLDER": "CARPETA DE SALIDA",
    "Select output folder…": "Seleccionar carpeta de salida…",
    "Browse…": "Examinar…",
    "Open Folder": "Abrir carpeta",
    "Enable physical units": "Activar unidades físicas",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "Escalar los valores de desplazamiento por el tamaño del píxel y mostrar unidades físicas en las etiquetas de la barra de color. La deformación es adimensional y no se ve afectada.",
    "/ pixel": "/ píxel",
    "Pixel size": "Tamaño del píxel",
    "fps": "fps",
    "Frame rate": "Fotogramas por segundo",
    "Data": "Datos",
    "Images": "Imágenes",
    "Animation": "Animación",
    "Report": "Informe",
    "FORMAT": "FORMATO",
    "NumPy Archive (.npz)": "Archivo NumPy (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV (por fotograma)",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ: un archivo por fotograma (predeterminado: un único archivo combinado)",
    "DISPLACEMENT": "DESPLAZAMIENTO",
    "Select:": "Seleccionar:",
    "STRAIN": "DEFORMACIÓN",
    "Run Compute Strain first.": "Ejecute primero «Calcular deformación».",
    "✓ Parameters file (JSON) always exported": "✓ El archivo de parámetros (JSON) siempre se exporta",
    "Export Data": "Exportar datos",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "Importar por lotes máscaras de región de interés",
    "Mask Folder:": "Carpeta de máscaras:",
    "(none)": "(ninguna)",
    "Browse...": "Examinar…",
    "Available Masks": "Máscaras disponibles",
    "Auto-Match by Name": "Coincidencia automática por nombre",
    "Match mask files to frames by number in filename":
        "Asociar archivos de máscara a fotogramas según el número del nombre de archivo",
    "Assign Sequential": "Asignar secuencialmente",
    "Assign masks to frames in order starting from frame 0":
        "Asignar máscaras a los fotogramas en orden desde el fotograma 0",
    "Frame Assignments": "Asignaciones de fotograma",
    "Frame": "Fotograma",
    "Image": "Imagen",
    "Mask": "Máscara",
    "Assign Selected ->": "Asignar selección ->",
    "Pair selected mask(s) with selected frame(s)":
        "Emparejar las máscaras seleccionadas con los fotogramas seleccionados",
    "Clear All": "Limpiar todo",

    # Canvas area / toolbar
    "Fit": "Ajustar",
    "Fit image to viewport": "Ajustar la imagen a la vista",
    "100%": "100%",
    "Zoom to 100% (1:1)": "Zoom al 100% (1:1)",
    "Zoom in": "Acercar",
    "–": "–",
    "Zoom out": "Alejar",
    "Show Grid": "Mostrar cuadrícula",
    "Show/hide computational mesh grid": "Mostrar/ocultar la cuadrícula de la malla",
    "Show Subset": "Mostrar subconjunto",
    "Show subset window on hover (requires Grid)":
        "Mostrar ventana del subconjunto al pasar el cursor (requiere cuadrícula)",
    "Placing Starting Points": "Colocando puntos de inicio",

    # Color range
    "Range": "Rango",
    "Auto": "Auto",
    "Min": "Mín",
    "Max": "Máx",

    # Field selector
    "Disp U": "Despl. U",
    "Disp V": "Despl. V",

    # Frame navigator
    "Play animation": "Reproducir animación",
    "▶": "▶",
    "Next frame": "Fotograma siguiente",
    "Playback speed": "Velocidad de reproducción",
    "FRAME 0/0": "FOTOGRAMA 0/0",
    "⏸": "⏸",
    "Pause animation": "Pausar animación",

    # Image list
    "Add": "Añadir",
    "Edit": "Editar",
    "Need": "Falta",

    # Mesh appearance
    "Mesh color": "Color de malla",
    "Click to choose mesh line color": "Haga clic para elegir el color de las líneas de la malla",
    "Line width": "Grosor de línea",

    # Physical units widget
    "Use physical units": "Usar unidades físicas",
    "Physical size of one image pixel": "Tamaño físico de un píxel de la imagen",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)":
        "Frecuencia de adquisición (usada para el campo de velocidad)",
    "Disp: px  Velocity: px/fr": "Despl.: px  Velocidad: px/fot.",

    # Advanced tuning
    "ADMM Iterations": "Iteraciones ADMM",
    "Number of ADMM alternating minimization cycles for AL-DIC.\n1 = single global pass (fastest), 3 = default,\n5+ = diminishing returns for most cases.":
        "Número de ciclos de minimización alternada ADMM para AL-DIC.\n1 = una única pasada global (más rápido), 3 = predeterminado,\n5+ = rendimiento decreciente en la mayoría de los casos.",
    "Only affects AL-DIC solver. Ignored by Local DIC.":
        "Solo afecta al solucionador AL-DIC. Local DIC lo ignora.",
    "Auto-expand FFT search on clipped peaks":
        "Expandir automáticamente la búsqueda FFT cuando los picos se recortan",
    "When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).\n\nOnly relevant for the FFT init-guess mode.":
        "Cuando el pico NCC alcanza el borde de la región de búsqueda, reintenta automáticamente con una región mayor (hasta la mitad del tamaño de la imagen, 6 reintentos con crecimiento ×2).\n\nSolo relevante para el modo de estimación inicial FFT.",

    # Canvas overlay
    "Mode": "Modo",
    "Init": "Inicial",
    "Accumulative": "Acumulativo",
    "Incremental": "Incremental",
    "Local DIC": "Local DIC",
    "Every Frame": "Cada fotograma",
    "Every N Frames": "Cada N fotogramas",
    "Custom Frames": "Fotogramas personalizados",
    "ADMM (%1 iter)": "ADMM (%1 iter.)",
    "FFT every frame": "FFT cada fotograma",
    "FFT every %1 fr": "FFT cada %1 fotogramas",
    "FFT": "FFT",

    # Log messages
    "Load images first.": "Cargue primero las imágenes.",
    "Building pipeline configuration...": "Construyendo configuración del flujo…",
    "Loading images...": "Cargando imágenes…",

    # Strain window
    "Strain Post-Processing": "Post-procesado de deformación",
    "Compute Strain": "Calcular deformación",
    "Starting…": "Iniciando…",
    "Complete": "Completado",
    "⚠ Params changed -- click Compute Strain": "⚠ Parámetros modificados — haga clic en «Calcular deformación»",
    "Unit: px/frame": "Unidad: px/fotograma",

    # Strain field selector
    "STRAIN PARAMETERS": "PARÁMETROS DE DEFORMACIÓN",

    # Strain param panel
    "Method": "Método",
    "Plane fitting": "Ajuste de plano",
    "FEM nodal": "FEM nodal",
    "VSG size": "Tamaño VSG",
    "Strain field smoothing": "Suavizado del campo de deformación",
    "Strain type": "Tipo de deformación",
    "Infinitesimal": "Infinitesimal",
    "Eulerian": "Euleriana",
    "Green-Lagrangian": "Green-Lagrange",
    "Gaussian smoothing of the strain field after computation.\nσ is the Gaussian kernel width; 'step' = DIC node spacing.\n  Light  (0.5 × step):  subtle, preserves fine features.\n  Medium (1 × step):    balanced, recommended for noisy data.\n  Strong (2 × step) ⚠:  aggressive, may blur real gradients.":
        "Suavizado gaussiano del campo de deformación tras el cálculo.\nσ es el ancho del núcleo gaussiano; «step» = espaciado de nodos DIC.\n  Ligero  (0,5 × step): sutil, conserva detalles finos.\n  Medio   (1 × step):   equilibrado, recomendado para datos ruidosos.\n  Fuerte  (2 × step) ⚠: agresivo, puede difuminar gradientes reales.",

    # Strain window export tooltip
    "Export displacement and strain results to NPZ / MAT / CSV / PNG":
        "Exportar resultados de desplazamiento y deformación a NPZ / MAT / CSV / PNG",
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
    print(f"[es] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
