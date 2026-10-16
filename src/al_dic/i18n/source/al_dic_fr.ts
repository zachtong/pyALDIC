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
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>crack</source>
        <translation>fissure</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>too few valid points</source>
        <translation>trop peu de points valides</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="79"/>
        <source>unreliable (strain edge trim)</source>
        <translation>non fiable (rognage des bords de la déformation)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>gauge endpoint lost</source>
        <translation>extrémité de jauge perdue</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="84"/>
        <source>not computed</source>
        <translation>non calculé</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="85"/>
        <source>no data</source>
        <translation>aucune donnée</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>Point</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>Ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>Rectangle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>Cercle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>Polygone</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="106"/>
        <source>Click once to place a point probe.</source>
        <translation>Cliquez une fois pour placer une sonde ponctuelle.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click twice: opposite corners.</source>
        <translation>Cliquez deux fois : coins opposés.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="116"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>Cliquez sur chaque sommet, puis double-cliquez pour fermer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>Afficher</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>Nom</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>Type</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>Couleur</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>Couleur…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>Remarque</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>Supprimer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>Tout effacer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="182"/>
        <source>Statistic:</source>
        <translation>Statistique :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>Moyenne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>Médiane</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="63"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>Maximum</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="64"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>Minimum</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="65"/>
        <source>Standard deviation</source>
        <translation>Écart-type</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="66"/>
        <source>Valid fraction</source>
        <translation>Fraction valide</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="50"/>
        <source>Crack opening</source>
        <translation>Ouverture de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="488"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>La déformation n&apos;a pas encore été calculée. Calculez-la dans l&apos;onglet « Champ de déformation » ou affichez un déplacement.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="532"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>Les grandeurs de jauge nécessitent une sonde linéaire.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="529"/>
        <source>No visible probe can show this quantity.</source>
        <translation>Aucune sonde visible ne peut afficher cette grandeur.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>no valid data: %1</source>
        <translation>aucune donnée valide : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>crack from frame %1</source>
        <translation>fissure dès l&apos;image %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="178"/>
        <source>endpoint lost from frame %1</source>
        <translation>extrémité perdue dès l&apos;image %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="181"/>
        <source>gaps: too few valid points</source>
        <translation>lacunes : trop peu de points valides</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="183"/>
        <source>gaps: unreliable strain</source>
        <translation>lacunes : déformation non fiable</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="189"/>
        <source>not plotted: gauges need a line</source>
        <translation>non tracée : les jauges nécessitent une ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="192"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>non tracée : un seul point n&apos;a ni dispersion ni couverture</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="194"/>
        <source>not plotted</source>
        <translation>non tracée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="510"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>Sonde « %1 » ajoutée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="596"/>
        <source>Clear All Probes</source>
        <translation>Effacer toutes les sondes</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="597"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>Supprimer toutes les sondes ? Cette action est irréversible.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>Point</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>Ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="479"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>Exécutez une analyse DIC pour tracer les sondes.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="96"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>Extensomètre virtuel</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="98"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>Jauge de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>Cliquez deux fois : début et fin. Une ligne est aussi un extensomètre virtuel et une jauge d&apos;ouverture de fissure.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click twice: center, then the edge.</source>
        <translation>Cliquez deux fois : centre, puis bord.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>Cliquez sur les deux points de mesure. Le graphique affiche alors la déformation entre eux.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="122"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>Cliquez sur un point de chaque côté de la fissure. Le graphique affiche alors son ouverture.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="132"/>
        <source>100%</source>
        <comment>Zoom button: one image pixel per screen pixel</comment>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>Afficher le champ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>Colore l&apos;image de référence avec le champ tracé à l&apos;image courante. Pour une lecture de jauge, le champ de l&apos;onglet « Champ de déformation » est affiché.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="180"/>
        <source>Plot:</source>
        <translation>Tracer :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>X axis:</source>
        <translation>Axe X :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>Strain as:</source>
        <translation>Déformation en :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="186"/>
        <source>Min. valid fraction:</source>
        <translation>Fraction valide min. :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>Une image reste vide lorsque moins de cette fraction des points d&apos;une ligne ou d&apos;une région est fiable. Évite une courbe qui reste lisse alors que son échantillon se réduit.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="194"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>Évolution temporelle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="196"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>La lecture de chaque sonde à chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="199"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>Le long de la ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="206"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>Kymographe</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="201"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>Le champ le long de la ligne sélectionnée à l&apos;image courante, sur les autres images en gris.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="208"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>Le champ le long de la ligne sélectionnée à chaque image : distance en fonction de l&apos;image, valeur en couleur.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="213"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>Contrainte–déformation</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="215"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>Contrainte (ou charge, sans A0) en fonction de la grandeur tracée, une courbe par sonde : contrainte-déformation avec un extensomètre, charge-ouverture avec une jauge de fissure. Nécessite des données de charge.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="226"/>
        <source>Other frames</source>
        <translation>Autres images</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>Tracer en pâle les profils des autres images derrière l&apos;image courante (au plus douze, régulièrement espacées).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="232"/>
        <source>Line data (CSV)…</source>
        <translation>Données de ligne (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Export</source>
        <translation>Exporter</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Load data…</source>
        <translation>Données de charge…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>Importer l&apos;enregistrement de charge d&apos;une machine d&apos;essai (CSV) pour tracer en fonction de la charge ou de la contrainte, et tracer des courbes contrainte-déformation.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="240"/>
        <source>Probe data (CSV)…</source>
        <translation>Données de sonde (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="242"/>
        <source>Chart image…</source>
        <translation>Image du graphique…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="244"/>
        <source>Copy chart</source>
        <translation>Copier le graphique</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="246"/>
        <source>Copy plotted data</source>
        <translation>Copier les données tracées</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="558"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>Importez l&apos;enregistrement de charge de la machine d&apos;essai via « Données de charge… » pour tracer des courbes contrainte-déformation.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="28"/>
        <source>Displacement U</source>
        <translation>Déplacement U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="29"/>
        <source>Displacement V</source>
        <translation>Déplacement V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="30"/>
        <source>Displacement magnitude</source>
        <translation>Norme du déplacement</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Extensometer strain</source>
        <translation>Déformation d&apos;extensomètre</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Extensometer true strain</source>
        <translation>Déformation vraie d&apos;extensomètre</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="49"/>
        <source>Elongation ΔL</source>
        <translation>Allongement ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="51"/>
        <source>Crack sliding</source>
        <translation>Glissement de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="52"/>
        <source>Crack opening magnitude</source>
        <translation>Norme de l&apos;ouverture de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="278"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="280"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="428"/>
        <source>Time (s)</source>
        <translation>Temps (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="282"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="430"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="573"/>
        <source>Load (N)</source>
        <translation>Charge (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="285"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="432"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="570"/>
        <source>Stress (MPa)</source>
        <translation>Contrainte (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="294"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>rapport</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="417"/>
        <source>Could not draw the field: %1</source>
        <translation>Impossible de dessiner le champ : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="134"/>
        <source>Esc cancels placement</source>
        <translation>Échap annule le placement</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="142"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>Faites glisser pour déplacer la sonde, ou une poignée pour la remodeler. Suppr la supprime ; F2 la renomme.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="484"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>Placez une sonde sur l&apos;image de référence pour commencer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="205"/>
        <source>Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.</source>
        <translation>Aucune donnée valide le long de %1 : sa déformation est rognée comme peu fiable près d&apos;un bord ou d&apos;un trou. Tracez un déplacement, ou rognez moins dans l&apos;onglet « Champ de déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="211"/>
        <source>Nothing valid along %1: a crack has consumed the material under it.</source>
        <translation>Aucune donnée valide le long de %1 : une fissure a consommé la matière en dessous.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="216"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>Aucune donnée valide le long de %1 : elle est hors de la zone mesurée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="609"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>Une vue de ligne affiche un champ. Choisissez un champ à tracer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="613"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>Placez ou sélectionnez une sonde linéaire pour voir le champ le long de celle-ci.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="631"/>
        <source>Distance along %1 (%2)</source>
        <translation>Distance le long de %1 (%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="640"/>
        <source>%1, frame %2</source>
        <translation>%1, image %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="157"/>
        <source>not plotted: off the measured area</source>
        <translation>non tracée : hors de la zone mesurée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="160"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>non tracée : une extrémité de la jauge est hors de la zone mesurée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>Export Probe Data</source>
        <translation>Exporter les données de sonde</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="657"/>
        <source>CSV Files</source>
        <translation>Fichiers CSV</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="439"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>Les données de charge ne peuvent pas être associées aux images : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="657"/>
        <location filename="../../gui/panels/analysis/tab.py" line="724"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="693"/>
        <source>Probe export failed: %1</source>
        <translation>Échec de l&apos;export de la sonde : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="695"/>
        <source>Probe data written to %1</source>
        <translation>Données de sonde écrites dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="703"/>
        <source>Export Line Data</source>
        <translation>Exporter les données de ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="714"/>
        <source>Line export failed: %1</source>
        <translation>Échec de l&apos;export de la ligne : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="716"/>
        <source>Line data written to %1</source>
        <translation>Données de ligne écrites dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="722"/>
        <source>SVG Images</source>
        <translation>Images SVG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="733"/>
        <source>Export Chart</source>
        <translation>Exporter le graphique</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="752"/>
        <source>Chart copied to the clipboard.</source>
        <translation>Graphique copié dans le presse-papiers.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="760"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>Données tracées copiées dans le presse-papiers.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="721"/>
        <source>PNG Images</source>
        <translation>Images PNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="723"/>
        <source>PDF Documents</source>
        <translation>Documents PDF</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="744"/>
        <source>Chart export failed: %1</source>
        <translation>Échec de l&apos;export du graphique : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="746"/>
        <source>Chart written to %1</source>
        <translation>Graphique écrit dans %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="1029"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>Exécutez d&apos;abord le DIC — aucun résultat de déplacement à post-traiter.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1185"/>
        <source>pyALDIC has hit an error</source>
        <translation>pyALDIC a rencontré une erreur</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1186"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>Une erreur inattendue s&apos;est produite. L&apos;application risque de ne plus fonctionner correctement ; il est recommandé d&apos;enregistrer la session et de redémarrer.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1196"/>
        <source>Details were written to %1</source>
        <translation>Les détails ont été écrits dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1302"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>Préparation des noyaux de calcul en arrière-plan. La première analyse après une nouvelle installation prend plus de temps que les suivantes.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1316"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>Noyaux de calcul prêts (%1 s).</translation>
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
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Import par lot des masques de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>Dossier de masques :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>(aucun)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>Masques disponibles</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>Correspondance auto par nom</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Associer les fichiers de masque aux images d&apos;après le numéro du nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>Attribuer séquentiellement</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Attribuer les masques aux images dans l&apos;ordre à partir de l&apos;image 0</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>Attributions d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>Fichier image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>Masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>Attribuer la sélection -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Associer les masques sélectionnés aux images sélectionnées</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>Tout effacer</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>Sélectionner le dossier de masques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>Échec de la lecture du fichier de masque.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Forme incompatible : %1×%2 (attendu %3×%4)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n masque a une taille incompatible et est désactivé.</numerusform>
            <numerusform>%n masques ont des tailles incompatibles et sont désactivés.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>Affectation non valide</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Une image ne peut avoir qu&apos;un seul masque. Sélectionnez exactement un masque, ou sélectionnez plusieurs images pour attribuer un masque à plusieurs.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1227"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1238"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1239"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1252"/>
        <source>Show Grid</source>
        <translation>Afficher la grille</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1253"/>
        <source>Show/hide computational mesh grid</source>
        <translation>Afficher/masquer la grille du maillage</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1256"/>
        <source>Show Subset</source>
        <translation>Afficher l&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1257"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>Afficher la fenêtre d&apos;imagette au survol (nécessite la grille)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1511"/>
        <source>Placing Starting Points</source>
        <translation>Placement des points de départ</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="45"/>
        <source>Mode</source>
        <translation>Mode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="46"/>
        <source>Solver</source>
        <translation>Solveur</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="47"/>
        <source>Init</source>
        <translation>Estimation initiale</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="97"/>
        <source>Accumulative</source>
        <translation>Cumulatif</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="98"/>
        <source>Incremental</source>
        <translation>Incrémental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="105"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="107"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM (%1 itér.)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>Starting Points</source>
        <translation>Points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="118"/>
        <source>FFT every frame</source>
        <translation>FFT chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="120"/>
        <source>FFT every %1 fr</source>
        <translation>FFT toutes les %1 images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="121"/>
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
        <location filename="../../gui/dialogs/export_dialog.py" line="868"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1025"/>
        <source>Auto</source>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="490"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1406"/>
        <source>Opacity</source>
        <extracomment>Whether an image sits behind the field. Independent of show_deformed: which frame to use is only a question once you show one at all. Fill when the background is hidden: &quot;white&quot;, &quot;black&quot; or &quot;transparent&quot;. &quot;Hide untracked regions&quot; of the window the dialog was opened from. Leave regions without a tracked node transparent (images). Leave regions without a tracked node transparent (animation). Fill used by images and animation alike when the background is hidden.</extracomment>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="492"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Opacité du champ (0 = transparent, 1 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="600"/>
        <source>All</source>
        <translation>Tous</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="602"/>
        <source>None</source>
        <translation>Aucun</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="631"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="655"/>
        <source>OUTPUT FOLDER</source>
        <translation>DOSSIER DE SORTIE</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Select output folder…</source>
        <translation>Sélectionner le dossier de sortie…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="667"/>
        <source>Browse…</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="672"/>
        <source>Open Folder</source>
        <translation>Ouvrir le dossier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="680"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="684"/>
        <source>Enable physical units</source>
        <translation>Activer les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="687"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Mettre à l&apos;échelle les valeurs de déplacement par la taille de pixel et afficher les unités physiques sur les étiquettes de la barre de couleur. La déformation est sans dimension et n&apos;est pas affectée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="704"/>
        <source>/ pixel</source>
        <translation>/ pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="706"/>
        <source>Pixel size</source>
        <translation>Taille de pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="721"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Frame rate</source>
        <translation>Fréquence d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="731"/>
        <source>Data</source>
        <translation>Données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="732"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="733"/>
        <source>Animation</source>
        <translation>Animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="734"/>
        <source>Report</source>
        <translation>Rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="736"/>
        <source>Preview &amp; Colorbar</source>
        <translation>Aperçu et barre de couleur</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>FORMAT</source>
        <translation>FORMAT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="769"/>
        <source>NumPy Archive (.npz)</source>
        <translation>Archive NumPy (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="771"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="773"/>
        <source>CSV (per frame)</source>
        <translation>CSV (par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="776"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ : un fichier par image (par défaut : un seul fichier fusionné)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <source>DISPLACEMENT</source>
        <translation>DÉPLACEMENT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="793"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="820"/>
        <source>Select:</source>
        <translation>Sélectionner :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="808"/>
        <source>STRAIN</source>
        <translation>DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Run Compute Strain first.</source>
        <translation>Exécutez d&apos;abord « Calculer la déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="838"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ Le fichier de paramètres (JSON) est toujours exporté</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="844"/>
        <source>Export Data</source>
        <translation>Exporter les données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="865"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1022"/>
        <source>Export</source>
        <translation>Exporter</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="866"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1023"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Field</source>
        <translation>Champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="867"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1024"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1379"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="869"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1026"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1391"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="870"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1027"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1398"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="889"/>
        <source>IMAGE SETTINGS</source>
        <translation>PARAMÈTRES D&apos;IMAGE</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="899"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1056"/>
        <source>Format</source>
        <translation>Format</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1064"/>
        <source>Full resolution</source>
        <translation>Résolution native</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="909"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>Limite le bord long de l&apos;image exportée (le plus grand de largeur/hauteur ; le ratio est conservé).
Le détail du champ est borné par le maillage, donc une limite plus petite est quasi sans perte,
mais bien plus légère et rapide à encoder. Plus petit = plus rapide. « Résolution native » conserve la taille native.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="916"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1071"/>
        <source>Resolution (long edge)</source>
        <translation>Résolution (bord long)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="950"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1108"/>
        <source>Deformed frame</source>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="951"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1109"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="955"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1113"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>Image déformée : le champ est tracé aux positions déplacées des nœuds (référence + déplacement), par-dessus la photo propre à chaque image.
Image de référence : tracé aux positions d&apos;origine des nœuds, par-dessus la première image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1119"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="966"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>Décochez pour n&apos;exporter que le champ, sans image de mouchetis derrière. Le remplissage se choisit dans l&apos;onglet « Aperçu et barre de couleur ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1066"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>Limite le bord long de l&apos;animation (le plus grand de largeur/hauteur).
Plus petit = plus rapide et bien plus léger. Fortement recommandé pour le GIF, dont la taille explose en résolution native.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="924"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>Qualité JPEG (plus élevée = fichier plus gros). Ignorée pour PNG/TIFF.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="927"/>
        <source>JPEG quality</source>
        <translation>Qualité JPEG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="936"/>
        <source>DPI</source>
        <translation>DPI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="938"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1097"/>
        <source>Include colorbar</source>
        <translation>Inclure la barre de couleur</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.
Les étiquettes se mettent à jour par image quand la plage auto est activée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="972"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1128"/>
        <source>Render as</source>
        <translation>Rendu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="973"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1129"/>
        <source>Hide untracked regions</source>
        <translation>Masquer les zones non suivies</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="976"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Leave areas without a tracked node transparent instead of filling them with values copied from distant nodes.</source>
        <translation>Laisser transparentes les zones sans nœud suivi au lieu de les remplir avec des valeurs copiées depuis des nœuds éloignés.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="998"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Cancel Export</source>
        <translation>Annuler l&apos;export</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1003"/>
        <source>Export Images</source>
        <translation>Exporter les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1046"/>
        <source>ANIMATION SETTINGS</source>
        <translation>PARAMÈTRES D&apos;ANIMATION</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1080"/>
        <source>FPS</source>
        <translation>FPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1090"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>Exporte une image sur N (1 = toutes les images). Plus élevé = plus rapide et plus léger,
mais plus saccadé. La durée de lecture est conservée (les FPS ci-dessus sont la fréquence d&apos;images avant décimation).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1095"/>
        <source>Frame step</source>
        <translation>Pas d&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1100"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.
Les étiquettes se mettent à jour par image quand la plage auto est activée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1122"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>Décochez pour n&apos;exporter que le champ, sans image de mouchetis derrière. GIF et MP4 ne peuvent pas stocker la transparence : un remplissage transparent est écrit en blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1156"/>
        <source>Export Animation</source>
        <translation>Exporter l&apos;animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1173"/>
        <source>CONTENT</source>
        <translation>CONTENU</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1175"/>
        <source>Parameter summary table</source>
        <translation>Tableau récapitulatif des paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1178"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Statistiques de champ (min/max/moyenne/écart-type par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1181"/>
        <source>Sample field images</source>
        <translation>Exemples d&apos;images de champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1188"/>
        <source>Sample every</source>
        <translation>Échantillonner toutes les</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1200"/>
        <source>FIELDS</source>
        <translation>CHAMPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1203"/>
        <source>Displacement:</source>
        <translation>Déplacement :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1218"/>
        <source>Strain:</source>
        <translation>Déformation :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1240"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Format : HTML (autonome, consultable dans n&apos;importe quel navigateur)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1247"/>
        <source>Generate Report</source>
        <translation>Générer le rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1269"/>
        <source>Open this tab to render a preview.</source>
        <translation>Ouvrez cet onglet pour générer un aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1285"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1297"/>
        <source>COLORBAR STYLE</source>
        <translation>STYLE DE BARRE DE COULEUR</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Right</source>
        <translation>Droite</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Left</source>
        <translation>Gauche</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1301"/>
        <source>Top</source>
        <translation>Haut</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1301"/>
        <source>Bottom</source>
        <translation>Bas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1304"/>
        <source>Position</source>
        <translation>Position</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1310"/>
        <source>Font size</source>
        <translation>Taille de police</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Font family</source>
        <translation>Police</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1324"/>
        <source>Bar thickness</source>
        <translation>Épaisseur de la barre</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1343"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1350"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1343"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1330"/>
        <source>Background</source>
        <translation>Arrière-plan</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1337"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>Ajoute une bordure vide autour du contenu exporté, en fraction du bord long (0 = aucune).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Margin</source>
        <translation>Marge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1346"/>
        <source>Margin color</source>
        <translation>Couleur de marge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1351"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1357"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.</source>
        <translation>Remplissage utilisé là où se serait trouvée l&apos;image de fond, quand « Afficher l&apos;image de fond » est décoché.
La transparence est conservée pour PNG et TIFF ; JPEG, GIF et MP4 n&apos;ont pas de canal alpha et reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1364"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1366"/>
        <source>Refresh preview</source>
        <translation>Actualiser l&apos;aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1373"/>
        <source>FIELD APPEARANCE</source>
        <translation>APPARENCE DU CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Range</source>
        <translation>Plage</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1408"/>
        <source>Apply to all fields</source>
        <translation>Appliquer à tous les champs</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1410"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Applique la palette, l&apos;opacité et la plage auto de ce champ à tous les champs activés (chaque champ garde ses propres min/max).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1533"/>
        <source>Preview failed: </source>
        <translation>Échec de l&apos;aperçu : </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1549"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Activez un champ dans l&apos;onglet Images pour l&apos;aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1562"/>
        <source>No data for this field/frame.</source>
        <translation>Aucune donnée pour ce champ/cette image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1640"/>
        <source>FRAME RANGE</source>
        <translation>PLAGE D&apos;IMAGES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1643"/>
        <source>All frames</source>
        <translation>Toutes les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1649"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>De</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1657"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>à</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1688"/>
        <source>Select Output Folder</source>
        <translation>Sélectionner le dossier de sortie</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1747"/>
        <source>Exported %1 files → %2</source>
        <translation>%1 fichiers exportés → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1756"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1886"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1971"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="2013"/>
        <source>Error: %1</source>
        <translation>Erreur : %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1781"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1914"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1859"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Rendu de %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1865"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1943"/>
        <source>Frame %1/%2</source>
        <translation>Image %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1875"/>
        <source>Exported %1 images → %2</source>
        <translation>%1 images exportées → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1953"/>
        <source>No animation was written. See the log for details.</source>
        <translation>Aucune animation n&apos;a été écrite. Consultez le journal pour plus de détails.</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="1963"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>%n animation exportée → %1</numerusform>
            <numerusform>%n animations exportées → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="2006"/>
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
        <location filename="../../gui/panels/canvas_area.py" line="1067"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Chargez d&apos;abord des images avant de dessiner une région d&apos;intérêt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1086"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Les trois points sont presque colinéaires — choisissez des points répartis sur le bord du cercle.</translation>
    </message>
</context>
<context>
    <name>ImageList</name>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>#</source>
        <comment>Image list column: frame index</comment>
        <translation>#</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="98"/>
        <source>Filename</source>
        <translation>Nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>Ajouter</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>Modifier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>Requis</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Importer la région d&apos;intérêt pour %n image</numerusform>
            <numerusform>Importer la région d&apos;intérêt pour %n images</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Effacer la région d&apos;intérêt (%1 avec région)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>Effacer la région d&apos;intérêt</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>Supprimer %n image</numerusform>
            <numerusform>Supprimer %n images</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="490"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>Sélectionner %n fichier de masque</numerusform>
            <numerusform>Sélectionner %n fichiers de masque</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="495"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="505"/>
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
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbors.

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
• Mouchetis bien texturé
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
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>Données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>Aucun fichier choisi.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>Choisir un fichier…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>Colonne de charge :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>Par le temps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>Par numéro d&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>Associer les lignes aux images :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>Colonne de temps :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>Le temps de la machine à l&apos;image de référence. Si la caméra a démarré 2 s après la machine, saisissez 2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>Décalage :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>Colonne d&apos;image :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>Numéro de la première image :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>Section initiale, pour la contrainte nominale F / A0 en MPa. Laisser à 0 pour la charge seule.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>Section A0 :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>Supprimer les données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>Fréquence d&apos;images de la caméra : %1 fps, issue des « Unités physiques ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>Définissez la fréquence d&apos;images de la caméra dans « Unités physiques » pour associer par le temps.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(sans nom)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>Aucune image ne tombe dans l&apos;enregistrement : vérifiez les colonnes et le décalage.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>Images avec une charge : %1 sur %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>Ouvrir les données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>Fichiers CSV</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>Impossible de lire %1 : %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="210"/>
        <source>&amp;File</source>
        <translation>Fichier</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="212"/>
        <source>Open Session…</source>
        <translation>Ouvrir une session…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Save Session…</source>
        <translation>Enregistrer la session…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="226"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>Associer les fichiers .aldic à pyALDIC…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="228"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>Enregistre .aldic pour qu&apos;un double-clic sur un fichier de session ouvre pyALDIC (utilisateur actuel uniquement, sans droits administrateur).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Quit</source>
        <translation>Quitter</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="242"/>
        <source>&amp;Settings</source>
        <translation>Paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="247"/>
        <source>Theme</source>
        <translation>Thème</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="252"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>Sombre</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="253"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>Clair</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="265"/>
        <source>Language</source>
        <translation>Langue</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="300"/>
        <source>Language changed</source>
        <translation>Langue modifiée</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="303"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>Langue définie sur %1. Veuillez redémarrer pyALDIC pour que tous les éléments prennent en compte la nouvelle langue.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Save Session</source>
        <translation>Enregistrer la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="319"/>
        <location filename="../../gui/app.py" line="368"/>
        <source>pyALDIC Session</source>
        <translation>Session pyALDIC</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="320"/>
        <location filename="../../gui/app.py" line="369"/>
        <location filename="../../gui/app.py" line="833"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="333"/>
        <source>large</source>
        <translation>volumineux</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="336"/>
        <source>Include Results?</source>
        <translation>Inclure les résultats ?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="338"/>
        <source>Include the computed results in this session?</source>
        <translation>Inclure les résultats calculés dans cette session ?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="341"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>Inclure les résultats (environ %1 non compressé) permet de rouvrir la session sans tout recalculer. Choisissez Non pour enregistrer un petit fichier de configuration seule, à partager.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="354"/>
        <source>Saving Session</source>
        <translation>Enregistrement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>Save Session Failed</source>
        <translation>Échec de l&apos;enregistrement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="366"/>
        <source>Open Session</source>
        <translation>Ouvrir une session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="381"/>
        <source>Loading Session</source>
        <translation>Chargement de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="384"/>
        <location filename="../../gui/app.py" line="419"/>
        <source>Open Session Failed</source>
        <translation>Échec de l&apos;ouverture de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="393"/>
        <source>Locate Session Images</source>
        <translation>Localiser les images de la session</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="396"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>Le dossier d&apos;images enregistré avec cette session est introuvable :
%1

Les résultats ont été restaurés. Pour afficher les images de fond, sélectionnez le dossier qui les contient désormais.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="404"/>
        <source>Select Image Folder</source>
        <translation>Sélectionner le dossier d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="480"/>
        <source>File Association Failed</source>
        <translation>Échec de l&apos;association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="481"/>
        <source>Could not register .aldic files: </source>
        <translation>Impossible d&apos;enregistrer les fichiers .aldic : </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="484"/>
        <source>File Association</source>
        <translation>Association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="485"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>Terminé. Un double-clic sur un fichier .aldic ouvrira désormais pyALDIC et restaurera cette session.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="800"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>L&apos;image %1 n&apos;a pas de région d&apos;intérêt propre — celle de l&apos;image 1 est utilisée pour le calcul. Passez à l&apos;image 1 pour la modifier, ou importez un masque pour donner à cette image la sienne.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="811"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>Aucune région d&apos;intérêt à enregistrer — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="818"/>
        <source>Region of Interest mask is empty.</source>
        <translation>Le masque de la région d&apos;intérêt est vide.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="830"/>
        <source>Save Region of Interest Mask</source>
        <translation>Enregistrer le masque de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="832"/>
        <source>PNG Images</source>
        <translation>Images PNG</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="841"/>
        <source>Mask saved to %1</source>
        <translation>Masque enregistré dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="850"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>Aucune région d&apos;intérêt à inverser — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="880"/>
        <location filename="../../gui/app.py" line="933"/>
        <source>Load images first.</source>
        <translation>Veuillez d&apos;abord charger des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="891"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>Définissez d&apos;abord une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="986"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Masque importé pour l&apos;image %1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="992"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>Import par lot : %n masque chargé</numerusform>
            <numerusform>Import par lot : %n masques chargés</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="1019"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Région d&apos;intérêt importée pour %n image</numerusform>
            <numerusform>Région d&apos;intérêt importée pour %n images</numerusform>
        </translation>
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
        <location filename="../../gui/widgets/param_panel.py" line="49"/>
        <source>Subset Size</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past a cap of about 28 characters of the label font, wrapped rather than squeezing the inputs.</extracomment>
        <translation>Taille d&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="55"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>Taille de la fenêtre d&apos;imagette IC-GN en pixels (nombre impair)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="62"/>
        <source>Subset Step</source>
        <translation>Pas d&apos;imagette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="65"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>Espacement des nœuds en pixels (doit être une puissance de 2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="73"/>
        <location filename="../../gui/widgets/param_panel.py" line="198"/>
        <location filename="../../gui/widgets/param_panel.py" line="210"/>
        <source>Search Range</source>
        <translation>Plage de recherche</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="88"/>
        <source>Refine Inner Boundary</source>
        <translation>Raffiner la limite interne</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="91"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>Raffiner localement le maillage le long des limites internes du masque
(trous à l&apos;intérieur de la région d&apos;intérêt). Utile pour les bords de bulles / vides.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="95"/>
        <source>Refine Outer Boundary</source>
        <translation>Raffiner la limite externe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="98"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>Raffiner localement le maillage le long de la limite externe de la région d&apos;intérêt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="113"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>Intensité du raffinement. Taille minimale d&apos;élément = max(2, subset_step / 2^level). S&apos;applique uniformément aux limites internes, externes ET aux zones peintes au pinceau. Les niveaux disponibles dépendent de la taille et du pas d&apos;imagette.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="119"/>
        <source>Refinement Level</source>
        <translation>Niveau de raffinement</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="179"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>Déplacement maximal par image détectable par la recherche FFT (pixels).
Définissez une valeur nettement supérieure au mouvement inter-images attendu.
Pour les grandes rotations en mode incrémental, cela doit couvrir :
  rayon × sin(angle par étape).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>Demi-largeur initiale (pixels) de la recherche NCC mono-point à chaque point de départ.
S&apos;étend automatiquement d&apos;un facteur 2 par tentative si le pic est tronqué, jusqu&apos;à la moitié de la taille de l&apos;image.
N&apos;affecte que l&apos;initialisation des points de départ ; les autres nœuds utilisent la propagation F-aware (pas de recherche par nœud).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="194"/>
        <location filename="../../gui/widgets/param_panel.py" line="210"/>
        <source>Starting Point Search</source>
        <translation>Recherche aux points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>Léger</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="251"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>Moyen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="252"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Fort</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="253"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Très fort</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="254"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>Ultra</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="260"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="282"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>taille min. d&apos;élément = %1 px  (subset_step=%2, level=%3)</translation>
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
        <translation>Fréquence d&apos;images d&apos;acquisition (utilisée pour le champ de vitesse)</translation>
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
        <location filename="../../gui/controllers/pipeline_controller.py" line="283"/>
        <source>Building pipeline configuration...</source>
        <translation>Construction de la configuration du pipeline…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="426"/>
        <source>Loading images...</source>
        <translation>Chargement des images…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="440"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  %1 images chargées, shape=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="452"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  Masque ROI : %1, %2 pixels (%3%)</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n image avec un masque ROI personnalisé</numerusform>
            <numerusform>  %n images avec des masques ROI personnalisés</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>Résultats reçus : %n image</numerusform>
            <numerusform>Résultats reçus : %n images</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  %n image, %1</numerusform>
            <numerusform>  %n images, %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  Étendue de la ROI : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  Raffinement de maillage : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>Erreur : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>Exécution annulée. Donnez aux images de référence listées leur propre région d&apos;intérêt, ou relancez et continuez avec la région de l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Cannot start analysis</source>
        <translation>Impossible de lancer l&apos;analyse</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="106"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>Au moins 2 images sont nécessaires. Chargez un dossier d&apos;images dans la section %1 avant de lancer la DIC.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="111"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>Définissez d&apos;abord une région d&apos;intérêt : dessinez-en ou importez-en une pour l&apos;image 1 dans la section %1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>La région d&apos;intérêt de l&apos;image 1 est vide (aucun pixel sélectionné). Redessinez-la dans la section %1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="121"/>
        <source>Region of Interest too small</source>
        <translation>Région d&apos;intérêt trop petite</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="127"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>La région d&apos;intérêt mesure %1×%2 px, mais le %3 actuel (%4) exige au moins %5×%5 px. Agrandissez la région d&apos;intérêt ou réduisez le %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="133"/>
        <source>Starting Points required</source>
        <translation>Points de départ requis</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="137"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>Placez au moins un point de départ sur le canevas avant de lancer l&apos;exécution : cliquez sur « %1 » ou « %2 » dans la section %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="143"/>
        <source>Invalid Starting Points</source>
        <translation>Points de départ non valides</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="147"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>Un ou plusieurs points de départ ne se trouvent dans aucune partie de la région d&apos;intérêt. Cliquez sur « %1 » dans la section %2, puis placez à nouveau les points de départ.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="155"/>
        <source>DIC analysis failed</source>
        <translation>Échec de l&apos;analyse DIC</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="158"/>
        <source>Failed to start analysis</source>
        <translation>Échec du lancement de l&apos;analyse</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="161"/>
        <source>Failed to process results</source>
        <translation>Échec du traitement des résultats</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="166"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>L&apos;analyse s&apos;est arrêtée sur une erreur d&apos;exécution :

%1

Consultez le journal de la console pour la trace complète.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="172"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

Consultez le journal de la console pour la trace complète.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>Couverture incomplète par les régions d&apos;intérêt</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="182"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>Cette exécution incrémentale compte %n image de référence sans région d&apos;intérêt propre : image %1.</numerusform>
            <numerusform>Cette exécution incrémentale compte %n images de référence sans région d&apos;intérêt propre : images %1.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 et %n autre</numerusform>
            <numerusform>%1 et %n autres</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="194"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>Une image de référence sans région d&apos;intérêt propre reprend telle quelle la région de l&apos;image 1 : rien ne déplace la région avec l&apos;éprouvette. Là où l&apos;éprouvette est sortie de cette région, la partie extérieure n&apos;est pas mesurée à partir d&apos;une telle image, et les nœuds qui s&apos;y trouvent cessent d&apos;être suivis ou voient leur déplacement copié depuis le nœud mesuré le plus proche. L&apos;exécution signale ces nœuds à la fin.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="204"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>Pour mesurer toute l&apos;éprouvette, donnez à chaque image de référence sa propre région d&apos;intérêt couvrant l&apos;éprouvette dans cette image (l&apos;import par lot permet d&apos;attribuer des régions d&apos;intérêt à de nombreuses images à la fois). Une image de référence dotée de sa propre région est mesurée sur une grille qui couvre cette région.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="211"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>Continuer avec la région de l&apos;image 1 ?</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="125"/>
        <source>Starting DIC analysis...</source>
        <translation>Démarrage de l&apos;analyse DIC…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="173"/>
        <source>Analysis complete in %1s</source>
        <translation>Analyse terminée en %1 s</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="166"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="181"/>
        <source>Analysis stopped by user.</source>
        <translation>Analyse arrêtée par l&apos;utilisateur.</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="63"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>Chargez d&apos;abord des images, puis dessinez une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="70"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Mode cumulatif&lt;/b&gt; — seule l&apos;image 1 a besoin d&apos;une région d&apos;intérêt. Toutes les images suivantes sont comparées directement à celle-ci.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="83"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — each frame is the reference for the next one. If the specimen moves, give each frame a Region of Interest that covers the specimen in that frame (Batch Import can assign many at once); reference frames without one reuse frame 1&apos;s region after you confirm.</source>
        <translation>&lt;b&gt;Incrémental, chaque image&lt;/b&gt; — chaque image sert de référence à la suivante. Si l&apos;éprouvette se déplace, donnez à chaque image une région d&apos;intérêt qui couvre l&apos;éprouvette dans cette image (l&apos;import par lot permet d&apos;en attribuer plusieurs à la fois) ; les images de référence sans région d&apos;intérêt propre reprennent celle de l&apos;image 1 après votre confirmation.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="103"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;Incrémental, toutes les %1 images&lt;/b&gt; — dessinez une région d&apos;intérêt sur les images : &lt;b&gt;%2&lt;/b&gt; (%3 images de référence au total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="117"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;Incrémental, personnalisé&lt;/b&gt; — aucune image de référence personnalisée définie. L&apos;image 1 sera la seule référence ; ajoutez d&apos;autres indices dans le champ Images de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="127"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;Incrémental, personnalisé&lt;/b&gt; — dessinez une région d&apos;intérêt sur les images : &lt;b&gt;%1&lt;/b&gt; (%2 images de référence au total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="135"/>
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
        <translation>Peindre des zones de raffinement de maillage supplémentaires au pinceau
(uniquement sur l&apos;image 1 — les points matériels sont automatiquement reportés sur les images suivantes)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>Le pinceau de raffinement n&apos;est disponible que sur l&apos;image 1. Passez à l&apos;image 1 pour peindre les zones de raffinement ; elles sont automatiquement reportées sur les images suivantes.</translation>
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
        <translation>Gommer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Effacer le pinceau</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>Polygone</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>Rectangle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>Cercle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>Cercle (3 points)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>Importer une image de masque</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>Exécuter l&apos;analyse DIC</translation>
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
        <location filename="../../gui/panels/right_sidebar.py" line="412"/>
        <source>Ready</source>
        <translation>Prêt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="414"/>
        <location filename="../../gui/panels/right_sidebar.py" line="514"/>
        <source>ELAPSED  %1</source>
        <translation>ÉCOULÉ  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="416"/>
        <location filename="../../gui/panels/right_sidebar.py" line="522"/>
        <location filename="../../gui/panels/right_sidebar.py" line="526"/>
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
        <location filename="../../gui/panels/right_sidebar.py" line="192"/>
        <source>Hide untracked regions</source>
        <translation>Masquer les zones non suivies</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="195"/>
        <source>Leave areas without a tracked node transparent, for example material that left the field of view, instead of filling them with values copied from distant nodes. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>Laisser transparentes les zones sans nœud suivi, par exemple le matériau sorti du champ de vision, au lieu de les remplir avec des valeurs copiées depuis des nœuds éloignés. S&apos;applique aussi aux images et animations exportées ; les fichiers de données y gardent toujours NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="205"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="210"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="229"/>
        <source>Opacity</source>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Opacité de la superposition (0 = transparent, 100 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="248"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="254"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="381"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Placez au moins un point de départ dans chaque région rouge avant de lancer l&apos;exécution (rouge = point de départ requis).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Image %2</translation>
    </message>
</context>
<context>
    <name>SeedErrorMessages</name>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="60"/>
        <source>Details: %1</source>
        <translation>Détails : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="63"/>
        <source>Starting Point match quality too low</source>
        <translation>Qualité de correspondance du point de départ trop faible</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="67"/>
        <source>A Starting Point could not be matched reliably between the reference frame and the deformed frame: the peak of its normalized cross-correlation (NCC) stayed below the acceptance threshold. The texture around it is probably too uniform.

Suggested actions:
- Move the Starting Point to a more textured area (visible speckle).
- Or choose “%1” in the %2 section.</source>
        <translation>Un point de départ n&apos;a pas pu être apparié de façon fiable entre l&apos;image de référence et l&apos;image déformée : le pic de sa corrélation croisée normalisée (NCC) est resté sous le seuil d&apos;acceptation. La texture autour de lui est probablement trop uniforme.

Actions suggérées :
- Déplacez le point de départ vers une zone plus texturée (mouchetis visible).
- Ou choisissez « %1 » dans la section %2.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="77"/>
        <source>Starting Point local solver did not converge</source>
        <translation>Le solveur local du point de départ n&apos;a pas convergé</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="81"/>
        <source>The IC-GN solve at a Starting Point reached its iteration limit without converging. Common causes: the point is too close to the edge of the Region of Interest, or the texture around it is too weak.

Suggested actions:
- Move the Starting Point inward, away from the edges of the Region of Interest.
- Try a smaller %1 (%2 section).
- Or choose “%3” in the %4 section.</source>
        <translation>La résolution IC-GN en un point de départ a atteint sa limite d&apos;itérations sans converger. Causes fréquentes : le point est trop près du bord de la région d&apos;intérêt, ou la texture autour de lui est trop faible.

Actions suggérées :
- Déplacez le point de départ vers l&apos;intérieur, loin des bords de la région d&apos;intérêt.
- Essayez une %1 plus petite (section %2).
- Ou choisissez « %3 » dans la section %4.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="94"/>
        <source>Starting Point flagged as unreliable</source>
        <translation>Point de départ jugé non fiable</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="98"/>
        <source>After the frame was solved, the displacement at a Starting Point was rejected as an outlier. The frame&apos;s field was propagated from the Starting Points, so the whole frame is unreliable.

Suggested actions:
- Move the Starting Point to a cleaner, well-textured area, away from the edges of the Region of Interest and from cracks.
- Or choose “%1” in the %2 section.</source>
        <translation>Après la résolution de l&apos;image, le déplacement en un point de départ a été rejeté comme valeur aberrante. Le champ de l&apos;image a été propagé depuis les points de départ, donc toute l&apos;image est peu fiable.

Actions suggérées :
- Déplacez le point de départ vers une zone plus propre et bien texturée, loin des bords de la région d&apos;intérêt et des fissures.
- Ou choisissez « %1 » dans la section %2.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="109"/>
        <source>Could not carry Starting Points to the new reference frame</source>
        <translation>Impossible de reporter les points de départ sur la nouvelle image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="113"/>
        <source>When the reference frame changed, the Starting Points could not be carried over to the new reference frame, and none could be placed there automatically. The material at the points has probably moved out of the Region of Interest of the new reference frame.

Suggested actions:
- Place Starting Points well inside the Region of Interest, away from its edges.
- Give each reference frame a Region of Interest that covers the specimen in that frame (%1 can assign Regions of Interest to many frames at once).</source>
        <translation>Lors du changement d&apos;image de référence, les points de départ n&apos;ont pas pu être reportés sur la nouvelle image de référence, et aucun n&apos;a pu y être placé automatiquement. La matière située aux points est probablement sortie de la région d&apos;intérêt de la nouvelle image de référence.

Actions suggérées :
- Placez les points de départ bien à l&apos;intérieur de la région d&apos;intérêt, loin de ses bords.
- Donnez à chaque image de référence une région d&apos;intérêt qui couvre l&apos;éprouvette dans cette image (« %1 » permet d&apos;attribuer des régions d&apos;intérêt à de nombreuses images à la fois).</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="126"/>
        <source>Region without a Starting Point</source>
        <translation>Région sans point de départ</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="130"/>
        <source>At least one region (a separate part of the Region of Interest) has no working Starting Point, even after placing one automatically was tried. The propagation cannot cross from one region to another, so every region needs its own Starting Point.

Suggested actions:
- Make sure every region shown in red has a Starting Point (“%1” in the %2 section).
- Or remove small separate pieces from the Region of Interest, so that fewer regions need a Starting Point.</source>
        <translation>Au moins une région (une partie séparée de la région d&apos;intérêt) n&apos;a aucun point de départ utilisable, même après une tentative de placement automatique. La propagation ne peut pas passer d&apos;une région à une autre : chaque région a donc besoin de son propre point de départ.

Actions suggérées :
- Vérifiez que chaque région affichée en rouge a un point de départ (« %1 » dans la section %2).
- Ou retirez de la région d&apos;intérêt les petites parties séparées, afin que moins de régions aient besoin d&apos;un point de départ.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="143"/>
        <source>Analysis stopped early</source>
        <translation>Analyse arrêtée prématurément</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="150"/>
        <source>Frame %1 could not be computed from the Starting Points, so the run stopped there; the frames completed before it, if any, are kept and shown. The details below name the cause.

Suggested actions:
- Place Starting Points on well-textured speckle, away from the edges of the Region of Interest and from cracks.
- Remove small separate pieces from the Regions of Interest: every piece needs a Starting Point, and none can be placed automatically on a piece without usable speckle.
- Or choose “%2” in the %3 section.</source>
        <translation>L&apos;image %1 n&apos;a pas pu être calculée à partir des points de départ, l&apos;exécution s&apos;est donc arrêtée là ; les images terminées avant elle, s&apos;il y en a, sont conservées et affichées. Les détails ci-dessous en indiquent la cause.

Actions suggérées :
- Placez les points de départ sur un mouchetis bien texturé, loin des bords de la région d&apos;intérêt et des fissures.
- Retirez des régions d&apos;intérêt les petites parties séparées : chaque partie a besoin d&apos;un point de départ, et aucun ne peut être placé automatiquement sur une partie sans mouchetis exploitable.
- Ou choisissez « %2 » dans la section %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="163"/>
        <source>Starting Points error</source>
        <translation>Erreur des points de départ</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/seed_error_messages.py" line="167"/>
        <source>Propagation from the Starting Points failed.

Suggested actions:
- Check the positions of the Starting Points.
- See the console log for details.</source>
        <translation>La propagation depuis les points de départ a échoué.

Actions suggérées :
- Vérifiez la position des points de départ.
- Consultez le journal de la console pour plus de détails.</translation>
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
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>Plane fitting</source>
        <translation>Ajustement de plan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="87"/>
        <source>FEM nodal</source>
        <translation>FEM nodal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Method</source>
        <translation>Méthode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="103"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>La taille VSG (Virtual Strain Gauge, jauge de déformation virtuelle) est le diamètre, en pixels, de la région circulaire autour de chaque nœud du maillage, utilisée pour ajuster un plan de déplacement local. La déformation est ensuite prise comme la pente de ce plan.

• VSG plus grande → déformation plus lisse, résolution spatiale plus faible.
• VSG plus petite → déformation plus fine, mais plus de bruit.
• Règle empirique : VSG ≥ 2 × pas d&apos;imagette + 1 (par défaut : 41 px).

Non utilisée quand Méthode = FEM nodal (l&apos;espacement du maillage fixe alors la taille).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="116"/>
        <source>VSG size</source>
        <translation>Taille VSG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="132"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>Nombre de nœuds de maillage par axe à l&apos;intérieur de la fenêtre VSG circulaire sur un maillage uniforme : 2 × floor(rayon VSG / espacement des nœuds) + 1. L&apos;ajustement de plan utilise tous les nœuds dans le rayon ; sur un maillage raffiné, ce nombre varie localement.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="163"/>
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
        <location filename="../../gui/widgets/strain_param_panel.py" line="175"/>
        <source>Trim low-confidence edges</source>
        <translation>Rogner les bords peu fiables</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Désactivé</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>Léger (σ = 0,5 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>Moyen (σ = 1 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="205"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>Fort (σ = 2 × step) ⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="211"/>
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
        <location filename="../../gui/widgets/strain_param_panel.py" line="217"/>
        <source>Strain field smoothing</source>
        <translation>Lissage du champ de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Infinitesimal</source>
        <translation>Infinitésimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Eulerian</source>
        <translation>Eulérien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="224"/>
        <source>Green-Lagrangian</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="226"/>
        <source>Strain type</source>
        <translation>Type de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="288"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>Rognés : %1 nœuds (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="364"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>Fenêtre de déformation ≈ %1×%2 nœuds</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="371"/>
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
        <location filename="../../gui/widgets/strain_viz_panel.py" line="155"/>
        <source>Hide untracked regions</source>
        <translation>Masquer les zones non suivies</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="158"/>
        <source>Leave areas without a tracked node transparent, for example material that left the field of view, instead of filling them with values copied from distant nodes. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>Laisser transparentes les zones sans nœud suivi, par exemple le matériau sorti du champ de vision, au lieu de les remplir avec des valeurs copiées depuis des nœuds éloignés. S&apos;applique aussi aux images et animations exportées ; les fichiers de données y gardent toujours NaN.</translation>
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
        <location filename="../../gui/strain_window.py" line="155"/>
        <source>Strain Post-Processing</source>
        <translation>Post-traitement de la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="204"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="214"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="272"/>
        <source>STRAIN PARAMETERS</source>
        <translation>PARAMÈTRES DE DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="291"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="295"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Annuler le calcul de déformation en cours. Le résultat de déformation précédent est conservé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Exporter les résultats de déplacement et de déformation en NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="340"/>
        <source>FIELD</source>
        <translation>CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="363"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="372"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="401"/>
        <source>Strain Field</source>
        <translation>Champ de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="403"/>
        <source>Analysis</source>
        <translation>Analyse</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="517"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Échec du calcul de déformation : %1 : %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="526"/>
        <location filename="../../gui/strain_window.py" line="588"/>
        <source>Strain computation complete.</source>
        <translation>Calcul de déformation terminé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="537"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Fenêtre de déformation : aucun résultat de déplacement à post-traiter.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="605"/>
        <source>Strain computation canceled.</source>
        <translation>Calcul de déformation annulé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="614"/>
        <source>Strain compute failed: %1</source>
        <translation>Échec du calcul de déformation : %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="621"/>
        <source>Strain Computation Failed</source>
        <translation>Échec du calcul de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="664"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Paramètres modifiés — cliquez sur « Calculer la déformation »</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="220"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="282"/>
        <source>Compute Strain</source>
        <translation>Calculer la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="304"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="548"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="568"/>
        <source>Canceling…</source>
        <translation>Annulation…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="584"/>
        <source>Complete</source>
        <translation>Terminé</translation>
    </message>
</context>
<context>
    <name>TrackingQuality</name>
    <message>
        <location filename="../../gui/controllers/tracking_messages.py" line="60"/>
        <source>Tracking Quality Warning</source>
        <translation>Avertissement sur la qualité du suivi</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/tracking_messages.py" line="64"/>
        <source>Some nodes could not be tracked reliably through the image sequence.</source>
        <translation>Certains nœuds n&apos;ont pas pu être suivis de façon fiable tout au long de la séquence d&apos;images.</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/tracking_messages.py" line="69"/>
        <source>In frame %1, %n node(s) were more than %2 px from the nearest node measured on reference frame %3. Their displacement was copied from that node and may be wrong.</source>
        <translation>
            <numerusform>Dans l&apos;image %1, %n nœud était à plus de %2 px du nœud le plus proche mesuré sur l&apos;image de référence %3. Son déplacement a été copié depuis ce nœud et peut être erroné.</numerusform>
            <numerusform>Dans l&apos;image %1, %n nœuds étaient à plus de %2 px du nœud le plus proche mesuré sur l&apos;image de référence %3. Leur déplacement a été copié depuis ce nœud et peut être erroné.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/tracking_messages.py" line="75"/>
        <source>%n node(s) (%1% of the tracked material) stopped being tracked by frame %2 because they landed on pixels outside the Region of Interest. Their displacement is not measured from that frame on (NaN in exported data).</source>
        <translation>
            <numerusform>À l&apos;image %2, %n nœud (%1% du matériau suivi) avait cessé d&apos;être suivi, car il était tombé sur des pixels hors de la région d&apos;intérêt. Son déplacement n&apos;est plus mesuré à partir de cette image (NaN dans les données exportées).</numerusform>
            <numerusform>À l&apos;image %2, %n nœuds (%1% du matériau suivi) avaient cessé d&apos;être suivis, car ils étaient tombés sur des pixels hors de la région d&apos;intérêt. Leur déplacement n&apos;est plus mesuré à partir de cette image (NaN dans les données exportées).</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/tracking_messages.py" line="82"/>
        <source>In frame %1 alone, %n node(s) (%2% of the tracked material) stopped being tracked because they landed on pixels outside the Region of Interest. Their displacement is not measured from that frame on (NaN in exported data).</source>
        <translation>
            <numerusform>Dans la seule image %1, %n nœud (%2% du matériau suivi) a cessé d&apos;être suivi, car il est tombé sur des pixels hors de la région d&apos;intérêt. Son déplacement n&apos;est plus mesuré à partir de cette image (NaN dans les données exportées).</numerusform>
            <numerusform>Dans la seule image %1, %n nœuds (%2% du matériau suivi) ont cessé d&apos;être suivis, car ils sont tombés sur des pixels hors de la région d&apos;intérêt. Leur déplacement n&apos;est plus mesuré à partir de cette image (NaN dans les données exportées).</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/tracking_messages.py" line="89"/>
        <source>%n node(s) moved too close to the image edge to be measured. Their displacement there was copied from the nearest measured node.</source>
        <translation>
            <numerusform>%n nœud s&apos;est trop approché du bord de l&apos;image pour être mesuré. Son déplacement y a été copié depuis le nœud mesuré le plus proche.</numerusform>
            <numerusform>%n nœuds se sont trop approchés du bord de l&apos;image pour être mesurés. Leur déplacement y a été copié depuis le nœud mesuré le plus proche.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/tracking_messages.py" line="95"/>
        <source>Measurement grid enlarged for %n reference frame(s) to cover their Regions of Interest (largest grid: %1 nodes on frame %2; frame 1 grid: %3 nodes).</source>
        <translation>
            <numerusform>Grille de mesure agrandie pour %n image de référence afin de couvrir sa région d&apos;intérêt (plus grande grille : %1 nœuds sur l&apos;image %2 ; grille de l&apos;image 1 : %3 nœuds).</numerusform>
            <numerusform>Grille de mesure agrandie pour %n images de référence afin de couvrir leurs régions d&apos;intérêt (plus grande grille : %1 nœuds sur l&apos;image %2 ; grille de l&apos;image 1 : %3 nœuds).</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/tracking_messages.py" line="101"/>
        <source>Suggested actions:
- Give every reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once).
- Keep the specimen inside the image for the whole sequence.
- If the specimen cracks or has holes, nodes lost along them are expected.</source>
        <translation>Actions suggérées :
- Donnez à chaque image de référence sa propre région d&apos;intérêt couvrant l&apos;éprouvette dans cette image (l&apos;import par lot permet d&apos;attribuer des régions d&apos;intérêt à de nombreuses images à la fois).
- Gardez l&apos;éprouvette dans l&apos;image pendant toute la séquence.
- Si l&apos;éprouvette se fissure ou comporte des trous, la perte de nœuds le long des fissures ou des trous est normale.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/tracking_messages.py" line="111"/>
        <source>Suggested actions:
- Check that each frame&apos;s Region of Interest covers the specimen in that frame.
- Keep the specimen inside the image for the whole sequence.
- If the specimen cracks or has holes, nodes lost along them are expected.</source>
        <translation>Actions suggérées :
- Vérifiez que la région d&apos;intérêt de chaque image couvre l&apos;éprouvette dans cette image.
- Gardez l&apos;éprouvette dans l&apos;image pendant toute la séquence.
- Si l&apos;éprouvette se fissure ou comporte des trous, la perte de nœuds le long des fissures ou des trous est normale.</translation>
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
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>Aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>(aucune image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>Image seule</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>Image + masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>Masque seul</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>Affichage :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="168"/>
        <source>Alpha:</source>
        <translation>Alpha :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="176"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Bleu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rouge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Vert</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Jaune</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>Couleur du masque :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>Aucun masque attribué</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>Image %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>Échec du chargement de l&apos;image</translation>
    </message>
</context>
</TS>
