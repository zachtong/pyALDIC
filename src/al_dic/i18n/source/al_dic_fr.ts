<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="fr" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>Itérations AL-DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>Nombre de cycles de raffinement global du solveur AL-DIC.
1 = passe unique (le plus rapide), 3 = par défaut,
5+ = rendement décroissant dans la plupart des cas.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>N&apos;affecte que le solveur AL-DIC. Ignoré par Local DIC.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>Étendre automatiquement la recherche FFT lors de pics tronqués</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>Lorsque le pic NCC atteint le bord de la zone de recherche, réessaie automatiquement avec une zone plus large (jusqu&apos;à la moitié de l&apos;image, 6 tentatives avec une croissance de 2×).

Uniquement pertinent pour le mode d&apos;estimation initiale FFT.</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="135"/>
        <location filename="../../gui/widgets/mpl_chart.py" line="182"/>
        <source>Frame</source>
        <extracomment>Shading for frames a probe could not measure.</extracomment>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="148"/>
        <source>Shaded frames: %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="184"/>
        <source>Distance along line (%1)</source>
        <translation>)</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="211"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <extracomment>Tool button label, tool token, and the probe kind it produces.</extracomment>
        <translation>t</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="212"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="213"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="214"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="215"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="218"/>
        <source>Click once to place a point probe.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="219"/>
        <source>Click twice: start and end of the gauge.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="220"/>
        <source>Click twice: opposite corners.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="221"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="223"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="231"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="232"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>m</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="233"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="234"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="236"/>
        <source>Colour…</source>
        <translation>…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="237"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="238"/>
        <source>Clear All</source>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="239"/>
        <source>Compare:</source>
        <translation>:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="240"/>
        <source>Field:</source>
        <translation>:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="241"/>
        <source>Statistic:</source>
        <translation>:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="242"/>
        <source>Minimum valid fraction:</source>
        <translation>:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="244"/>
        <source>A frame is left blank when fewer than this fraction of the probe&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="248"/>
        <source>Export CSV…</source>
        <translation>…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="249"/>
        <source>Export Chart…</source>
        <translation>…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="250"/>
        <source>Esc cancels placement</source>
        <translation>t</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="265"/>
        <source>Point probes</source>
        <translation>s</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="266"/>
        <source>Line probes</source>
        <translation>s</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="267"/>
        <source>Region probes</source>
        <translation>n</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="294"/>
        <source>Value</source>
        <comment>Statistic: the sample itself, for a point probe</comment>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="295"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="296"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="297"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>m</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="298"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>m</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="299"/>
        <source>Standard deviation</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="300"/>
        <source>Valid fraction</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="301"/>
        <source>Engineering strain</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="302"/>
        <source>Crack opening</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="332"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="391"/>
        <source>Clear All Probes</source>
        <translation>s</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="392"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="432"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>t</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="433"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="434"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="473"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="484"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="502"/>
        <source>This statistic does not apply here.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="538"/>
        <location filename="../../gui/panels/analysis_tab.py" line="543"/>
        <source>Export Probe Data</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="539"/>
        <source>There is nothing to export yet.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="544"/>
        <source>CSV Files</source>
        <translation>V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="544"/>
        <location filename="../../gui/panels/analysis_tab.py" line="570"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="557"/>
        <source>Probe export failed: %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="562"/>
        <source>Probe data written to %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="567"/>
        <source>Export Chart</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="568"/>
        <source>PNG Images</source>
        <translation>G</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="569"/>
        <source>PDF Documents</source>
        <translation>F</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="578"/>
        <source>Chart export failed: %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="583"/>
        <source>Chart written to %1</source>
        <translation>1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="983"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>Région d&apos;intérêt importée pour %n images</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>Exécutez d&apos;abord le DIC — aucun résultat de déplacement à post-traiter.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1094"/>
        <source>pyALDIC has hit an error</source>
        <translation>r</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1095"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1105"/>
        <source>Details were written to %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1209"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1223"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>.</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>Ajuster la plage de couleurs à la plage de données de chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>Fixe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>Conserver les bornes Min/Max manuelles pour toutes les images</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Import par lot des masques de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>Dossier de masques :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>(aucun)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>Masques disponibles</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>Correspondance auto par nom</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Associer les fichiers de masque aux images d&apos;après le numéro du nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>Attribuer séquentiellement</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Attribuer les masques aux images dans l&apos;ordre à partir de l&apos;image 0</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>Attributions d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>Masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>Attribuer la sélection -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Associer les masques sélectionnés aux images sélectionnées</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>Tout effacer</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>Sélectionner le dossier de masques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>Échec de la lecture du fichier de masque.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Forme incompatible : %1×%2 (attendu %3×%4)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>%n masque(s) ont des tailles incompatibles et sont désactivés.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>Affectation non valide</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Une image ne peut avoir qu&apos;un seul masque. Sélectionnez exactement un masque, ou sélectionnez plusieurs images pour attribuer un masque à plusieurs.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1226"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1233"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>Afficher la grille</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>Afficher/masquer la grille du maillage</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>Afficher l&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>Afficher la fenêtre d&apos;imagette au survol (nécessite la grille)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>Placement des points de départ</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Mode</source>
        <translation>Mode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Solver</source>
        <translation>Solveur</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="43"/>
        <source>Init</source>
        <translation>Initial</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>Cumulatif</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="94"/>
        <source>Incremental</source>
        <translation>Incrémental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="101"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM (%1 itér.)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="110"/>
        <source>Starting Points</source>
        <translation>Points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>FFT chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>FFT toutes les %1 images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="117"/>
        <source>FFT</source>
        <translation>FFT</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>Plage</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="859"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1008"/>
        <source>Auto</source>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="481"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1382"/>
        <source>Opacity</source>
        <extracomment>Whether an image sits behind the field. Independent of show_deformed: which frame to use is only a question once you show one at all. Fill when the background is hidden: &quot;white&quot;, &quot;black&quot; or &quot;transparent&quot;. Fill used by images and animation alike when the background is hidden.</extracomment>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Opacité du champ (0 = transparent, 1 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>Tout</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>Aucun</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>DOSSIER DE SORTIE</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>Sélectionner le dossier de sortie…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>Ouvrir le dossier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>Activer les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Mettre à l&apos;échelle les valeurs de déplacement par la taille du pixel et afficher les unités physiques sur les étiquettes de la barre de couleurs. La déformation est sans dimension et n&apos;est pas affectée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="695"/>
        <source>/ pixel</source>
        <translation>/ pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="697"/>
        <source>Pixel size</source>
        <translation>Taille du pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="712"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="714"/>
        <source>Frame rate</source>
        <translation>Cadence</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>Données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>Animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>Rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>Aperçu et barre de couleur</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>FORMAT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>Archive NumPy (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV (par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ : un fichier par image (par défaut : un seul fichier fusionné)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="775"/>
        <source>DISPLACEMENT</source>
        <translation>DÉPLACEMENT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Select:</source>
        <translation>Sélectionner :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>Exécutez d&apos;abord « Calculer la déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ Le fichier de paramètres (JSON) est toujours exporté</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>Exporter les données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>Exporter</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>Champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="860"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1009"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1367"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="861"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1010"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1374"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="880"/>
        <source>IMAGE SETTINGS</source>
        <translation>PARAMÈTRES D&apos;IMAGE</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="890"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1039"/>
        <source>Format</source>
        <translation>Format</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="898"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1047"/>
        <source>Full resolution</source>
        <translation>Résolution native</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>Limite le bord long de l&apos;image exportée (le plus grand de largeur/hauteur ; le ratio est conservé).
Le détail du champ est borné par le maillage, donc une limite plus petite est quasi sans perte,
mais bien plus légère et rapide à encoder. Plus petit = plus rapide. « Résolution native » conserve la taille native.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>Résolution (bord long)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>Image déformée : le champ est tracé aux positions déplacées des nœuds (référence + déplacement), par-dessus la photo propre à chaque image.
Image de référence : tracé aux positions d&apos;origine des nœuds, par-dessus la première image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>Décochez pour n&apos;exporter que le champ, sans image de mouchetis derrière. Le remplissage se choisit dans l&apos;onglet Preview &amp; Colorbar.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>Limite le bord long de l&apos;animation (le plus grand de largeur/hauteur).
Plus petit = plus rapide et bien plus léger. Fortement recommandé pour le GIF, dont la taille explose en résolution native.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>Qualité JPEG (plus élevée = fichier plus gros). Ignorée pour PNG/TIFF.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>Qualité JPEG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="927"/>
        <source>DPI</source>
        <translation>DPI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="929"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1080"/>
        <source>Include colorbar</source>
        <translation>Inclure la barre de couleur</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.
Les étiquettes se mettent à jour par image quand la plage auto est activée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>Rendu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>Annuler l&apos;export</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>Exporter les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>PARAMÈTRES D&apos;ANIMATION</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1063"/>
        <source>FPS</source>
        <translation>FPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1073"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>Exporte une image sur N (1 = toutes les images). Plus élevé = plus rapide et plus léger,
mais plus saccadé. La durée de lecture est conservée (les FPS ci-dessus sont le débit avant décimation).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>Pas d&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.
Les étiquettes se mettent à jour par image quand la plage auto est activée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>Décochez pour n&apos;exporter que le champ, sans image de mouchetis derrière. GIF et MP4 ne peuvent pas stocker la transparence : un remplissage transparent est écrit en blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>Exporter l&apos;animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>CONTENU</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>Tableau récapitulatif des paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Statistiques de champ (min/max/moyenne/écart-type par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>Exemples d&apos;images de champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1164"/>
        <source>Sample every</source>
        <translation>Échantillonner toutes les</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1170"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>CHAMPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>Déplacement :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>Déformation :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Format : HTML (autonome, consultable dans n&apos;importe quel navigateur)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>Générer le rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>Ouvrez cet onglet pour générer un aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>STYLE DE BARRE DE COULEUR</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Right</source>
        <translation>Droite</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Left</source>
        <translation>Gauche</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Top</source>
        <translation>Haut</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Bottom</source>
        <translation>Bas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1280"/>
        <source>Position</source>
        <translation>Position</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1286"/>
        <source>Font size</source>
        <translation>Taille de police</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>Police</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>Épaisseur de la barre</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1326"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1325"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1306"/>
        <source>Background</source>
        <translation>Arrière-plan</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1313"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>Ajoute une bordure vide autour du contenu exporté, en fraction du bord long (0 = aucune).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>Marge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>Couleur de marge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1333"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.</source>
        <translation>Remplissage utilisé là où se serait trouvée l&apos;image de fond, quand « Afficher l&apos;image de fond » est décoché.
La transparence est conservée pour PNG et TIFF ; JPEG, GIF et MP4 n&apos;ont pas de canal alpha et reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>Actualiser l&apos;aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>APPARENCE DU CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>Plage</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>Appliquer à tous les champs</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Applique la colormap, l&apos;opacité et l&apos;auto-plage de ce champ à tous les champs activés (chaque champ garde ses propres min/max).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>Échec de l&apos;aperçu : </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Activez un champ dans l&apos;onglet Images pour l&apos;aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>Aucune donnée pour ce champ/cette image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>PLAGE D&apos;IMAGES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>Toutes les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>De</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1619"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>à</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1650"/>
        <source>Select Output Folder</source>
        <translation>Sélectionner le dossier de sortie</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>%1 fichiers exportés → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>Erreur : %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Rendu de %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>Image %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>%1 images exportées → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>Rapport enregistré → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="18"/>
        <source>Disp U</source>
        <translation>Dépl. U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Disp V</source>
        <translation>Dépl. V</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>Lire l&apos;animation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="73"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="169"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="80"/>
        <source>Next frame</source>
        <translation>Image suivante</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>Vitesse de lecture</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>IMAGE 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>Pause</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>IMAGE %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Chargez d&apos;abord des images avant de dessiner une région d&apos;intérêt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Les trois points sont presque colinéaires — choisissez des points répartis sur le bord du cercle.</translation>
    </message>
</context>
<context>
    <name>ImageList</name>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="95"/>
        <source>#</source>
        <comment>Image list column: frame index</comment>
        <translation>#</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="96"/>
        <source>Filename</source>
        <translation>Nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="190"/>
        <location filename="../../gui/widgets/image_list.py" line="248"/>
        <source>Add</source>
        <translation>Ajouter</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="242"/>
        <source>Edit</source>
        <translation>Modifier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>Requis</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>Importer la région d&apos;intérêt pour %n images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Effacer la région d&apos;intérêt (%1 avec région)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>Effacer la région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>Supprimer %n images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%1 fichiers sélectionnés pour %2 images — le nombre doit correspondre</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>Points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>Placez quelques points ; pyALDIC initialise chacun avec une NCC mono-point et propage le champ le long des voisins du maillage.

Idéal pour :
• Grands déplacements inter-images (&gt; 50 px)
• Champs discontinus (fissures, bandes de cisaillement)
• Scénarios où la FFT choisit de mauvais pics

Placés automatiquement par région lorsque vous dessinez ou modifiez une ROI.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>Placer les points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>Entrer en mode placement sur le canevas. Clic gauche pour ajouter, clic droit pour supprimer, Échap ou nouveau clic pour sortir.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>Placement automatique</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>Remplit les régions vides avec le nœud ayant la meilleure NCC. Les points de départ existants sont conservés.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>Supprimer tous les points de départ. Plus rapide que de cliquer droit sur chacun.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 régions prêtes</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT (corrélation croisée)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>Corrélation croisée normalisée sur la grille complète. Robuste dans le rayon de recherche ; la recherche s&apos;étend automatiquement lorsque les pics sont tronqués.

Idéal pour :
• Mouvements lisses petits à modérés
• Speckle bien texturé
• Aucune configuration spéciale requise

Le coût augmente avec le rayon de recherche, les très grands déplacements deviennent donc lents.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>Toutes les</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>Exécute la FFT toutes les N images. N = 1 signifie FFT à chaque image (le plus sûr, le plus lent). N &gt; 1 utilise un démarrage à chaud entre les réinitialisations pour limiter la propagation d&apos;erreurs à N images.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>(N=1 = chaque image)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>Uniquement à la mise à jour de la référence (incrémental seulement)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>Exécute la FFT à chaque changement d&apos;image de référence ; démarrage à chaud dans chaque segment. Valeur par défaut typique pour le mode incrémental.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>Utilise le déplacement convergé de l&apos;image précédente comme estimation initiale. Aucune corrélation croisée n&apos;est exécutée.

Idéal pour :
• Très petits mouvements inter-images (quelques pixels)
• Option la plus rapide lorsque le mouvement est lisse

Les erreurs peuvent s&apos;accumuler sur les séquences longues. Préférez FFT ou les points de départ sur des données bruitées ou lorsque le mouvement est plus important.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>Placement… (cliquez pour sortir)</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>IMAGES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="185"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>Tri naturel (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="188"/>
        <source>Sort by embedded numbers: image1, image2, …, image10
Default (unchecked): lexicographic — best for zero-padded names</source>
        <translation>Tri par numéros intégrés : image1, image2, …, image10
Par défaut (non coché) : lexicographique — idéal pour les noms avec zéros de remplissage</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>TYPE DE FLUX</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>ESTIMATION INITIALE</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>RÉGION D&apos;INTÉRÊT</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>PARAMÈTRES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>AVANCÉ</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>Fichier</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>Ouvrir une session…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>Enregistrer la session…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>Associer les fichiers .aldic à pyALDIC…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>Enregistre .aldic pour qu&apos;un double-clic sur un fichier de session ouvre pyALDIC (utilisateur actuel uniquement, sans droits administrateur).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>Quitter</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>Paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>Langue</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>Langue modifiée</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>Langue définie sur %1. Veuillez redémarrer pyALDIC pour que tous les éléments prennent en compte la nouvelle langue.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>Enregistrer la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>Session pyALDIC</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>volumineux</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>Inclure les résultats ?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>Inclure les résultats calculés dans cette session ?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>Inclure les résultats (environ %1 non compressé) permet de rouvrir la session sans tout recalculer. Choisissez Non pour enregistrer un petit fichier de configuration seule, à partager.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>Enregistrement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>Échec de l&apos;enregistrement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>Ouvrir une session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>Chargement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>Échec de l&apos;ouverture de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>Localiser les images de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>Le dossier d&apos;images enregistré avec cette session est introuvable :
%1

Les résultats ont été restaurés. Pour afficher les images d&apos;arrière-plan, sélectionnez le dossier qui les contient désormais.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>Sélectionner le dossier d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>Échec de l&apos;association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>Impossible d&apos;enregistrer les fichiers .aldic : </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>Association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>Terminé. Un double-clic sur un fichier .aldic ouvrira désormais pyALDIC et restaurera cette session.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>L&apos;image %1 n&apos;a pas de région d&apos;intérêt propre — celle de l&apos;image 1 est utilisée pour le calcul. Passez à l&apos;image 1 pour la modifier, ou importez un masque pour donner à cette image la sienne.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>Aucune région d&apos;intérêt à enregistrer — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>Le masque de la région d&apos;intérêt est vide.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>t</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>G</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>Masque enregistré dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>Aucune région d&apos;intérêt à inverser — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>Veuillez d&apos;abord charger des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>Définissez d&apos;abord une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Masque importé pour l&apos;image %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>Import par lot : %n masque(s) chargé(s)</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>Couleur du maillage</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>Cliquer pour choisir la couleur des lignes du maillage</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>Épaisseur</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="37"/>
        <source>Subset Size</source>
        <translation>Taille d&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="43"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>Taille de la fenêtre d&apos;imagette IC-GN en pixels (nombre impair)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="50"/>
        <source>Subset Step</source>
        <translation>Pas d&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="53"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>Espacement des nœuds en pixels (doit être une puissance de 2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Search Range</source>
        <translation>Plage de recherche</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Refine Inner Boundary</source>
        <translation>Raffiner la limite interne</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="79"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>Raffiner localement le maillage le long des limites internes du masque
(trous à l&apos;intérieur de la région d&apos;intérêt). Utile pour les bords de bulles / vides.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="83"/>
        <source>Refine Outer Boundary</source>
        <translation>Raffiner la limite externe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>Raffiner localement le maillage le long de la limite externe de la région d&apos;intérêt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>Intensité du raffinage. Taille minimale d&apos;élément = max(2, subset_step / 2^niveau). S&apos;applique uniformément aux limites internes, externes ET aux zones peintes au pinceau. Les niveaux disponibles dépendent de la taille et du pas d&apos;imagette.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>Niveau de raffinage</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>Déplacement maximal par image détectable par la recherche FFT (pixels).
Définissez une valeur nettement supérieure au mouvement inter-image attendu.
Pour les grandes rotations en mode incrémental, cela doit couvrir :
  rayon × sin(angle par étape).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>Demi-largeur initiale (pixels) de la recherche NCC mono-point à chaque point de départ.
S&apos;étend automatiquement d&apos;un facteur 2 par tentative si le pic est tronqué, jusqu&apos;à la moitié de la taille de l&apos;image.
N&apos;affecte que l&apos;initialisation des points de départ ; les autres nœuds utilisent la propagation F-aware (pas de recherche par nœud).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Initial Seed Search</source>
        <translation>Recherche initiale du germe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="218"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>Léger</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="219"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>Moyen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="220"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Fort</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="221"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Très fort</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="222"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>Ultra</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="228"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>taille min. d&apos;élément = %1 px  (subset_step=%2, niveau=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>Utiliser les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>Taille physique d&apos;un pixel de l&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="80"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>Pixel size</source>
        <translation>Taille de pixel</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>Cadence d&apos;acquisition (utilisée pour le champ de vitesse)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>Fréquence d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>Dépl. : %1  Vitesse : %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>Dépl. : px  Vitesse : px/im</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>Construction de la configuration du pipeline…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>Chargement des images…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  %1 images chargées, forme=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  Masque ROI : %1, %2 pixels (%3%)</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>Exécution annulée : définissez les régions d&apos;intérêt par image pour les images de référence manquantes, ou acceptez le masque hérité de l&apos;image 1 au prochain lancement.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n images avec des masques ROI personnalisés</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>Résultats reçus : %n images</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>Démarrage de l&apos;analyse DIC…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>Analyse terminée en %1 s</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>Analyse arrêtée par l&apos;utilisateur.</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>Chargez d&apos;abord des images, puis dessinez une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Mode cumulatif&lt;/b&gt; — seule l&apos;image 1 a besoin d&apos;une région d&apos;intérêt. Toutes les images suivantes sont comparées directement à celle-ci.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;Incrémental, chaque image&lt;/b&gt; — l&apos;image 1 a besoin d&apos;une région d&apos;intérêt. Elle est automatiquement propagée à chaque image suivante (pas de dessin par image requis).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;Incrémental, toutes les %1 images&lt;/b&gt; — dessinez une région d&apos;intérêt sur les images : &lt;b&gt;%2&lt;/b&gt; (%3 images de référence au total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;Incrémental, personnalisé&lt;/b&gt; — aucune image de référence personnalisée définie. L&apos;image 1 sera la seule référence ; ajoutez d&apos;autres indices dans le champ Images de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;Incrémental, personnalisé&lt;/b&gt; — dessinez une région d&apos;intérêt sur les images : &lt;b&gt;%1&lt;/b&gt; (%2 images de référence au total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>Dessinez une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ Ajouter</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Ajouter une région à la région d&apos;intérêt (Polygone / Rectangle / Cercle)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>Découper</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Découper une région de la région d&apos;intérêt (Polygone / Rectangle / Cercle)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ Raffiner</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>Peindre des zones de raffinage de maillage supplémentaires au pinceau
(uniquement sur l&apos;image 1 — les points matériels sont automatiquement reportés sur les images suivantes)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>Le pinceau de raffinage n&apos;est disponible que sur l&apos;image 1. Passez à l&apos;image 1 pour peindre les zones de raffinage ; elles sont automatiquement reportées sur les images suivantes.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>Importer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>Importer le masque depuis un fichier image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>Import par lot</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>Importer par lot des fichiers de masques pour plusieurs images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>Enregistrer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>Enregistrer le masque actuel en PNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>Inverser</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>Inverser le masque de la région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>Effacer tous les masques de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>Rayon</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>Peindre</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Effacer le pinceau</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="256"/>
        <source>Circle (3-point)</source>
        <translation>Cercle (3 points)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="350"/>
        <source>Import Mask Image</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>Lancer l&apos;analyse DIC</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>Annuler l&apos;analyse en cours. Les images déjà calculées sont conservées, ce qui permet de consulter ou d&apos;exporter le résultat partiel.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>Ouvrir la fenêtre de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>Calculer et visualiser la déformation dans une fenêtre de post-traitement séparée. Nécessite des résultats de déplacement d&apos;une exécution terminée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>PROGRESSION</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>Prêt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>ÉCOULÉ  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>RESTANT  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>Afficher sur</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Tracer le champ aux positions déformées des nœuds, ou à leurs positions dans l&apos;image de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Décochez pour n&apos;afficher que le champ, sans image de mouchetis derrière.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="178"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="183"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Ce qui remplace l&apos;image lorsqu&apos;elle est masquée. À l&apos;export, la transparence est conservée pour PNG et TIFF ; les autres formats reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>Palette de couleurs</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Opacité de la superposition (0 = transparent, 100 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Placez au moins un point de départ dans chaque région rouge avant de lancer l&apos;exécution (rouge = point de départ requis).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Image %2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="150"/>
        <source>DISPLACEMENT</source>
        <translation>DÉPLACEMENT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="161"/>
        <source>STRAIN</source>
        <translation>DÉFORMATION</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>Lire l&apos;animation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="88"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="215"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="95"/>
        <source>Next frame</source>
        <translation>Image suivante</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>Vitesse de lecture</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>IMAGE 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>Pause</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>IMAGE %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>Ajustement de plan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>FEM nodal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Method</source>
        <translation>Méthode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="102"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>La taille VSG (Virtual Strain Gauge, jauge de déformation virtuelle) est le diamètre, en pixels, de la région circulaire autour de chaque nœud du maillage, utilisée pour ajuster un plan de déplacement local. La déformation est ensuite prise comme la pente de ce plan.

• VSG plus grande → déformation plus lisse, résolution spatiale plus faible.
• VSG plus petite → déformation plus fine, mais plus de bruit.
• Règle empirique : VSG ≥ 2 × pas de subset + 1 (par défaut : 41 px).

Non utilisée quand Méthode = FEM nodal (l&apos;espacement du maillage fixe alors la taille).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>Taille VSG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>Nombre de nœuds de maillage par axe à l&apos;intérieur de la fenêtre VSG circulaire sur un maillage uniforme : 2 × floor(rayon VSG / espacement des nœuds) + 1. L&apos;ajustement de plan utilise tous les nœuds dans le rayon ; sur un maillage raffiné, ce nombre varie localement.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>Masque la déformation peu fiable aux bords de la ROI / des trous, là où la fenêtre VSG franchit la frontière et où l&apos;ajustement de plan local devient unilatéral et peu fiable.

• Coefficient × rayon VSG = largeur de la bande de bord rognée.
• 0.00 = conserver tous les nœuds (aucun rognage).
• 0.70 = recommandé (rogne là où l&apos;erreur de bord augmente fortement).
• 1.00 = le plus strict (rogne tout nœud dont la fenêtre touche le bord).

Ne s&apos;applique que lorsque Méthode = Ajustement de plan.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>Rogner les bords peu fiables</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Désactivé</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>Léger (σ = 0,5 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>Moyen (σ = 1 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>Fort (σ = 2 × step) ⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>Lissage gaussien du champ de déformation après calcul.
σ est la largeur du noyau gaussien ; « step » = espacement des nœuds DIC.
  Léger   (0,5 × step) : subtil, préserve les détails fins.
  Moyen   (1 × step) :   équilibré, recommandé pour des données bruitées.
  Fort    (2 × step) ⚠ : agressif, peut flouter les vrais gradients.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>Lissage du champ de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>Infinitésimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>Eulérien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>Type de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>Rognés : %1 nœuds (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>Fenêtre de déformation ≈ %1×%2 nœuds</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ Rayon VSG (%1 px) &lt; espacement des nœuds DIC (%2 px) ; l&apos;ajustement de plan échouera. Utilisez VSG ≥ %3 px ou passez la Méthode en FEM nodal.</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Tracer le champ aux positions déformées des nœuds, ou à leurs positions dans l&apos;image de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>Afficher sur</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Décochez pour n&apos;afficher que le champ, sans image de mouchetis derrière.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="82"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="83"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="87"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Ce qui remplace l&apos;image lorsqu&apos;elle est masquée. À l&apos;export, la transparence est conservée pour PNG et TIFF ; les autres formats reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>Arrière-plan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>Plage</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="122"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="134"/>
        <source>Opacity</source>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="143"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>Remplir les bords rognés (affichage uniquement)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>Réinterpole la bande de déformation rognée aux bords à partir de nœuds intérieurs fiables. Affecte l&apos;affichage à l&apos;écran et les images/animations exportées ; les fichiers de données exportés conservent toujours le bord rogné en NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>Bords</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="154"/>
        <source>Strain Post-Processing</source>
        <translation>Post-traitement de la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="202"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="209"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="213"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="271"/>
        <source>STRAIN PARAMETERS</source>
        <translation>PARAMÈTRES DE DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="290"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="294"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Annuler le calcul de déformation en cours. Le résultat de déformation précédent est conservé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="306"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Exporter les résultats de déplacement et de déformation en NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="339"/>
        <source>FIELD</source>
        <translation>CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="348"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="361"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="370"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="399"/>
        <source>Strain Field</source>
        <translation>n</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="401"/>
        <source>Analysis</source>
        <translation>e</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="476"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Échec du calcul de déformation : %1 : %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="484"/>
        <location filename="../../gui/strain_window.py" line="543"/>
        <source>Strain computation complete.</source>
        <translation>Calcul de déformation terminé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="495"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Fenêtre de déformation : aucun résultat de déplacement à post-traiter.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="526"/>
        <source>Cancelling…</source>
        <translation>Annulation…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="560"/>
        <source>Strain computation cancelled.</source>
        <translation>Calcul de déformation annulé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="569"/>
        <source>Strain compute failed: %1</source>
        <translation>Échec du calcul de déformation : %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="576"/>
        <source>Strain Computation Failed</source>
        <translation>Échec du calcul de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="615"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Paramètres modifiés — cliquez sur « Calculer la déformation »</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="281"/>
        <source>Compute Strain</source>
        <translation>Calculer la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="303"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="506"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="539"/>
        <source>Complete</source>
        <translation>Terminé</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>Utiliser les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>Unité : px/image</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="51"/>
        <source>Incremental</source>
        <translation>Incrémental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="52"/>
        <source>Accumulative</source>
        <translation>Cumulatif</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>Incrémental : chaque image est comparée à l&apos;image de référence précédente.
Adapté aux grandes déformations cumulées, requis pour les grandes rotations.

Cumulatif : chaque image est comparée à l&apos;image 1.
Précis uniquement pour les petites déformations monotones.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>Mode de suivi</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="75"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="78"/>
        <source>Local DIC: Independent subset matching (IC-GN). Fast,
preserves sharp local features. Best for small
deformations or high-quality images.

AL-DIC: Augmented Lagrangian with global FEM
regularization. Enforces displacement compatibility
between subsets. Best for large deformations, noisy
images, or when strain accuracy matters.</source>
        <translation>Local DIC : Appariement d&apos;imagettes indépendant (IC-GN). Rapide,
préserve les détails locaux. Idéal pour les petites
déformations ou les images de haute qualité.

AL-DIC : Lagrangien augmenté avec régularisation
FEM globale. Impose la compatibilité des déplacements
entre imagettes. Idéal pour les grandes déformations, les images
bruitées ou lorsque la précision de la déformation est importante.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>Solveur</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>Chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>Toutes les N images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>Images personnalisées</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>Quand l&apos;image de référence est rafraîchie lors du suivi incrémental.
Chaque image : réinitialiser la référence à chaque image (plus petit déplacement par étape,
plus robuste pour les grandes déformations).
Toutes les N images : réinitialiser toutes les N images (compromis vitesse/robustesse).
Images personnalisées : liste d&apos;indices définis par l&apos;utilisateur.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>Mise à jour de la référence</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>Mettre à jour la référence toutes les N images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>Intervalle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>Indices d&apos;images séparés par des virgules pour les images de référence (base 0)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>Images de référence</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>Déposez un dossier d&apos;images
ou Parcourir</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>Sélectionner le dossier d&apos;images</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>Aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>(aucune image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>Image seule</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>Image + masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>Masque seul</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>Affichage :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>Alpha :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Bleu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rouge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Vert</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Jaune</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>Couleur du masque :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>Aucun masque attribué</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>Image %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>Échec du chargement de l&apos;image</translation>
    </message>
</context>
</TS>
