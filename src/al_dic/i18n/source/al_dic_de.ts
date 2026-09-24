<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="de" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>AL-DIC-Iterationen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>Anzahl globaler Verfeinerungszyklen für den AL-DIC-Löser.
1 = einmaliger Durchlauf (schnellste), 3 = Standard,
5+ = abnehmender Ertrag in den meisten Fällen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>Betrifft nur den AL-DIC-Löser. Wird von Local DIC ignoriert.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>FFT-Suche bei abgeschnittenen Peaks automatisch erweitern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>Wenn der NCC-Peak den Rand des Suchbereichs erreicht, wird automatisch mit einem größeren Bereich wiederholt (bis zur halben Bildgröße, 6 Versuche mit 2-facher Vergrößerung).

Nur relevant für den FFT-Anfangsschätzungsmodus.</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="70"/>
        <source>crack</source>
        <translation>Riss</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="72"/>
        <source>too few valid points</source>
        <translation>zu wenige gültige Punkte</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="74"/>
        <source>unreliable (strain edge trim)</source>
        <translation>unzuverlässig (Dehnungs-Randbeschnitt)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>gauge endpoint lost</source>
        <translation>Endpunkt der Messstrecke verloren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="79"/>
        <source>not computed</source>
        <translation>nicht berechnet</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>no data</source>
        <translation>keine Daten</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>Punkt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>Rechteck</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>Kreis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>Polygon</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="106"/>
        <source>Click once to place a point probe.</source>
        <translation>Einmal klicken, um eine Punktsonde zu setzen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click twice: opposite corners.</source>
        <translation>Zweimal klicken: gegenüberliegende Ecken.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>Zweimal klicken: Mittelpunkt, dann Rand.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="116"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>Jeden Eckpunkt anklicken, dann per Doppelklick schließen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="87"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>Anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>Name</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>Typ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>Farbe</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>Hinweis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="98"/>
        <source>Colour…</source>
        <translation>Farbe…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="99"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Clear All</source>
        <translation>Alle löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Statistic:</source>
        <translation>Statistik:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>Mittelwert</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>Median</translation>
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
        <translation>Standardabweichung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="66"/>
        <source>Valid fraction</source>
        <translation>Gültiger Anteil</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="50"/>
        <source>Crack opening</source>
        <translation>Rissöffnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="487"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>Die Dehnung wurde noch nicht berechnet. Berechnen Sie sie im Reiter „Dehnungsfeld“ oder stellen Sie eine Verschiebung dar.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="531"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>Messstreckengrößen benötigen eine Liniensonde.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="528"/>
        <source>No visible probe can show this quantity.</source>
        <translation>Keine sichtbare Sonde kann diese Größe anzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>no valid data: %1</source>
        <translation>keine gültigen Daten: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>crack from frame %1</source>
        <translation>Riss ab Bild %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="178"/>
        <source>endpoint lost from frame %1</source>
        <translation>Endpunkt ab Bild %1 verloren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="181"/>
        <source>gaps: too few valid points</source>
        <translation>Lücken: zu wenige gültige Punkte</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="183"/>
        <source>gaps: unreliable strain</source>
        <translation>Lücken: unzuverlässige Dehnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="189"/>
        <source>not plotted: gauges need a line</source>
        <translation>nicht dargestellt: Messstrecken benötigen eine Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="192"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>nicht dargestellt: ein einzelner Punkt hat weder Streuung noch Abdeckung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="194"/>
        <source>not plotted</source>
        <translation>nicht dargestellt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="508"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>Sonde &apos;%1&apos; hinzugefügt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="594"/>
        <source>Clear All Probes</source>
        <translation>Alle Sonden löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="595"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>Alle Sonden löschen? Das lässt sich nicht rückgängig machen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="108"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>Punkt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="109"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="478"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>Führen Sie eine DIC-Analyse aus, um Sonden darzustellen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="96"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>Virtueller Dehnungsaufnehmer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="98"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>Riss-Messstrecke</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>Zweimal klicken: Anfang und Ende. Eine Linie ist auch ein virtueller Dehnungsaufnehmer und eine Rissöffnungs-Messstrecke.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>Klicken Sie auf die beiden Messpunkte. Das Diagramm zeigt dann die Dehnung zwischen ihnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="122"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>Klicken Sie auf je einen Punkt auf beiden Seiten des Risses. Das Diagramm zeigt dann, wie weit er sich öffnet.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="122"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="129"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="130"/>
        <source>100%</source>
        <comment>Zoom button: one image pixel per screen pixel</comment>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="133"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="134"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Show field</source>
        <comment>Analysis canvas: colour the image by the field</comment>
        <translation>Feld anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Colour the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>Färbt das Referenzbild mit dem dargestellten Feld im aktuellen Bild ein. Bei einer Messstreckengröße wird das Feld des Reiters „Dehnungsfeld“ angezeigt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="179"/>
        <source>Plot:</source>
        <translation>Darstellen:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="182"/>
        <source>X axis:</source>
        <translation>X-Achse:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Strain as:</source>
        <translation>Dehnung als:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Min. valid fraction:</source>
        <translation>Min. gültiger Anteil:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="186"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>Ein Bild bleibt leer, wenn weniger als dieser Anteil der Punkte einer Linie oder eines Bereichs zuverlässig ist. Verhindert eine Kurve, die glatt bleibt, während ihre Stichprobe schrumpft.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="193"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>Zeitverlauf</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>Der Messwert jeder Sonde in jedem Bild.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="198"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>Entlang der Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>The field along the selected line at the current frame, over the other frames in grey.</source>
        <translation>Das Feld entlang der ausgewählten Linie im aktuellen Bild, dahinter die übrigen Bilder in Grau.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="205"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>Kymogramm</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>The field along the selected line at every frame: distance against frame, value as colour.</source>
        <translation>Das Feld entlang der ausgewählten Linie in jedem Bild: Abstand über Bild, Wert als Farbe.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="212"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>Spannung–Dehnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>Spannung (ohne A0 die Kraft) über der dargestellten Größe, eine Kurve je Sonde: Spannungs-Dehnungs-Kurve mit einem Dehnungsaufnehmer, Kraft über Rissöffnung mit einer Riss-Messstrecke. Benötigt Kraftdaten.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="225"/>
        <source>Other frames</source>
        <translation>Übrige Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="226"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>Die Profile der übrigen Bilder blass hinter dem aktuellen zeichnen (höchstens zwölf, gleichmäßig verteilt).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="231"/>
        <source>Line data (CSV)…</source>
        <translation>Liniendaten (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="232"/>
        <source>Export</source>
        <translation>Exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Load data…</source>
        <translation>Kraftdaten…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>Kraftaufzeichnung einer Prüfmaschine (CSV) importieren, um über Kraft oder Spannung darzustellen und Spannungs-Dehnungs-Kurven zu zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="239"/>
        <source>Probe data (CSV)…</source>
        <translation>Sondendaten (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Chart image…</source>
        <translation>Diagrammbild…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Copy chart</source>
        <translation>Diagramm kopieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy plotted data</source>
        <translation>Dargestellte Daten kopieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="557"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>Importieren Sie die Kraftaufzeichnung der Prüfmaschine über „Kraftdaten…“, um Spannungs-Dehnungs-Kurven zu zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="28"/>
        <source>Displacement U</source>
        <translation>Verschiebung U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="29"/>
        <source>Displacement V</source>
        <translation>Verschiebung V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="30"/>
        <source>Displacement magnitude</source>
        <translation>Verschiebungsbetrag</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Extensometer strain</source>
        <translation>Dehnungsaufnehmer-Dehnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Extensometer true strain</source>
        <translation>Wahre Dehnung (Dehnungsaufnehmer)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="49"/>
        <source>Elongation ΔL</source>
        <translation>Verlängerung ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="51"/>
        <source>Crack sliding</source>
        <translation>Rissgleiten</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="52"/>
        <source>Crack opening magnitude</source>
        <translation>Betrag der Rissöffnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="277"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="432"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="427"/>
        <source>Time (s)</source>
        <translation>Zeit (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="572"/>
        <source>Load (N)</source>
        <translation>Kraft (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="284"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="569"/>
        <source>Stress (MPa)</source>
        <translation>Spannung (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="293"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>Verhältnis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="417"/>
        <source>Could not draw the field: %1</source>
        <translation>Das Feld konnte nicht gezeichnet werden: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="134"/>
        <source>Esc cancels placement</source>
        <translation>Esc bricht das Setzen ab</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="142"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>Ziehen verschiebt die Sonde, Ziehen an einem Griff ändert ihre Form. Entf löscht sie, F2 benennt sie um.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="483"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>Setzen Sie eine Sonde auf das Referenzbild, um zu beginnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="205"/>
        <source>Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.</source>
        <translation>Keine gültigen Daten entlang %1: Die Dehnung wird nahe einem Rand oder Loch als wenig zuverlässig beschnitten. Stellen Sie eine Verschiebung dar oder beschneiden Sie im Reiter „Dehnungsfeld“ weniger.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="211"/>
        <source>Nothing valid along %1: a crack has consumed the material under it.</source>
        <translation>Keine gültigen Daten entlang %1: Ein Riss hat das Material darunter aufgezehrt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="216"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>Keine gültigen Daten entlang %1: Die Linie liegt außerhalb des Messbereichs.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="608"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>Eine Linienansicht zeigt ein Feld. Wählen Sie ein Feld zum Darstellen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="612"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>Setzen oder wählen Sie eine Liniensonde, um das Feld entlang der Linie zu sehen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="630"/>
        <source>Distance along %1 (%2)</source>
        <translation>Abstand entlang %1 (%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="639"/>
        <source>%1, frame %2</source>
        <translation>%1, Bild %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="157"/>
        <source>not plotted: off the measured area</source>
        <translation>nicht dargestellt: außerhalb des Messbereichs</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="160"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>nicht dargestellt: ein Messstreckenende liegt außerhalb des Messbereichs</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="662"/>
        <source>Export Probe Data</source>
        <translation>Sondendaten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <source>CSV Files</source>
        <translation>CSV-Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="439"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>Die Kraftdaten lassen sich den Bildern nicht zuordnen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <location filename="../../gui/panels/analysis/tab.py" line="722"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="691"/>
        <source>Probe export failed: %1</source>
        <translation>Sondenexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="693"/>
        <source>Probe data written to %1</source>
        <translation>Sondendaten nach %1 geschrieben</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Export Line Data</source>
        <translation>Liniendaten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="712"/>
        <source>Line export failed: %1</source>
        <translation>Linienexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="714"/>
        <source>Line data written to %1</source>
        <translation>Liniendaten nach %1 geschrieben</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>SVG Images</source>
        <translation>SVG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="731"/>
        <source>Export Chart</source>
        <translation>Diagramm exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart copied to the clipboard.</source>
        <translation>Diagramm in die Zwischenablage kopiert.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="758"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>Dargestellte Daten in die Zwischenablage kopiert.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="719"/>
        <source>PNG Images</source>
        <translation>PNG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="721"/>
        <source>PDF Documents</source>
        <translation>PDF-Dokumente</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="742"/>
        <source>Chart export failed: %1</source>
        <translation>Diagrammexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="744"/>
        <source>Chart written to %1</source>
        <translation>Diagramm nach %1 geschrieben</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="983"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>Region of Interest für %n Bilder importiert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>DIC zuerst ausführen — keine Verschiebungs-Ergebnisse zur Nachbearbeitung.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1094"/>
        <source>pyALDIC has hit an error</source>
        <translation>In pyALDIC ist ein Fehler aufgetreten</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1095"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>Ein unerwarteter Fehler ist aufgetreten. Die Anwendung verhält sich möglicherweise nicht mehr korrekt; es wird empfohlen, die Sitzung zu speichern und neu zu starten.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1105"/>
        <source>Details were written to %1</source>
        <translation>Details wurden nach %1 geschrieben</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1209"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>Rechenkernel werden im Hintergrund vorbereitet. Die erste Analyse einer neuen Installation dauert länger als die folgenden.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1223"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>Rechenkernel bereit (%1 s).</translation>
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
        <translation>Farbbereich an den Datenbereich jedes Bildes anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>Fest</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>Manuelle Min/Max-Grenzen für alle Bilder beibehalten</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Region-of-Interest-Masken stapelweise importieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>Maskenordner:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>(keine)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>Durchsuchen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>Verfügbare Masken</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>Automatisch nach Name zuordnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Maskendateien anhand der Zahl im Dateinamen den Bildern zuordnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>Sequentiell zuweisen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Masken den Bildern der Reihe nach ab Bild 0 zuweisen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>Bildzuweisungen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>Bilddatei</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>Auswahl zuweisen -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Ausgewählte Maske(n) mit ausgewähltem/ausgewählten Bild(ern) koppeln</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>Alle löschen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>Maskenordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>Maskendatei konnte nicht gelesen werden.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Größe stimmt nicht: %1×%2 (erwartet %3×%4)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>%n Maske(n) haben abweichende Größen und sind deaktiviert.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>Ungültige Zuordnung</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Ein Bild kann nur eine Maske haben. Wählen Sie genau eine Maske aus oder wählen Sie mehrere Bilder, um eine Maske mehreren zuzuweisen.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1226"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1233"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>Gitter anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>Berechnungsnetz ein-/ausblenden</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>Subset anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>Subset-Fenster beim Überfahren anzeigen (erfordert Gitter)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>Startpunkte werden platziert</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Mode</source>
        <translation>Modus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Solver</source>
        <translation>Löser</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="43"/>
        <source>Init</source>
        <translation>Anfangsschätzung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>Akkumulativ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="94"/>
        <source>Incremental</source>
        <translation>Inkrementell</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="101"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM (%1 Iter.)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="110"/>
        <source>Starting Points</source>
        <translation>Startpunkte</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>FFT jedes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>FFT alle %1 Bilder</translation>
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
        <translation>Bereich</translation>
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
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Feld-Deckkraft (0 = transparent, 1 = vollständig deckend)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>Alle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>Keine</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>AUSGABEORDNER</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>Ausgabeordner auswählen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>Durchsuchen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>Ordner öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>Physikalische Einheiten aktivieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Verschiebungswerte mit der Pixelgröße skalieren und physikalische Einheiten auf den Farbleistenbeschriftungen anzeigen. Dehnung ist dimensionslos und wird nicht beeinflusst.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="695"/>
        <source>/ pixel</source>
        <translation>/ Pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="697"/>
        <source>Pixel size</source>
        <translation>Pixelgröße</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="712"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="714"/>
        <source>Frame rate</source>
        <translation>Bildrate</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>Daten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>Animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>Bericht</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>Vorschau &amp; Farbleiste</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>FORMAT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy-Archiv (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV (pro Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ: eine Datei pro Bild (Standard: eine zusammengeführte Datei)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="775"/>
        <source>DISPLACEMENT</source>
        <translation>VERSCHIEBUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Select:</source>
        <translation>Auswählen:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>DEHNUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>Zuerst „Dehnung berechnen“ ausführen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ Parameterdatei (JSON) wird immer exportiert</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>Daten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>Exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>Feld</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
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
        <translation>BILDEINSTELLUNGEN</translation>
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
        <translation>Volle Auflösung</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>Begrenzt die lange Kante des exportierten Bildes (das Größere von Breite/Höhe; Seitenverhältnis bleibt erhalten).
Die Felddetails sind durch das Netz begrenzt, daher ist eine kleinere Grenze nahezu verlustfrei,
aber viel kleiner und schneller zu kodieren. Kleiner = schneller. „Volle Auflösung“ behält die native Größe bei.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>Auflösung (lange Kante)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>Verformtes Bild: Das Feld wird an den verschobenen Knotenpositionen (Referenz + Verschiebung) über dem jeweils eigenen Foto jedes Bildes gezeichnet.
Referenzbild: an den ursprünglichen Knotenpositionen über dem ersten Bild gezeichnet.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter zu exportieren. Die Füllung wird im Reiter „Vorschau &amp; Farbleiste“ gewählt.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>Begrenzt die lange Kante der Animation (das Größere von Breite/Höhe).
Kleiner = schneller und viel kleiner. Dringend empfohlen für GIF, dessen Größe bei nativer Auflösung explodiert.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>JPEG-Qualität (höher = größere Datei). Wird für PNG/TIFF ignoriert.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>JPEG-Qualität</translation>
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
        <translation>Farbleiste einfügen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.
Die Beschriftungen aktualisieren sich pro Bild, wenn Auto aktiv ist.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>Darstellen als</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>Export abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>Bilder exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>ANIMATIONSEINSTELLUNGEN</translation>
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
        <translation>Jedes N-te Bild exportieren (1 = jedes Bild). Höher = schneller und kleiner,
wirkt aber ruckeliger. Die Abspieldauer bleibt erhalten (die FPS oben sind die Rate vor der Dezimierung).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>Bildschritt</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.
Die Beschriftungen aktualisieren sich pro Bild, wenn Auto aktiv ist.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter zu exportieren. GIF und MP4 können keine Transparenz speichern; eine transparente Füllung wird als Weiß geschrieben.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>Animation exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>INHALT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>Parameter-Übersichtstabelle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Feldstatistik (min/max/Mittelwert/Stdabw. pro Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>Beispiel-Feldbilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1164"/>
        <source>Sample every</source>
        <translation>Alle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1170"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>FELDER</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>Verschiebung:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>Dehnung:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Format: HTML (eigenständig, in jedem Browser anzeigbar)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>Bericht erstellen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>Diesen Reiter öffnen, um eine Vorschau zu rendern.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>FARBLEISTEN-STIL</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Right</source>
        <translation>Rechts</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Left</source>
        <translation>Links</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Top</source>
        <translation>Oben</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Bottom</source>
        <translation>Unten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1280"/>
        <source>Position</source>
        <translation>Position</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1286"/>
        <source>Font size</source>
        <translation>Schriftgröße</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>Schriftart</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>Balkendicke</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1326"/>
        <source>Black</source>
        <translation>Schwarz</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1325"/>
        <source>White</source>
        <translation>Weiß</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1306"/>
        <source>Background</source>
        <translation>Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1313"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>Fügt einen leeren Rand um den exportierten Inhalt hinzu, als Anteil der langen Kante (0 = keiner).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>Rand</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>Randfarbe</translation>
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
        <translation>Füllung für den Bereich, in dem sonst das Hintergrundbild läge, wenn „Hintergrundbild anzeigen“ aus ist.
Bei PNG und TIFF bleibt die Transparenz erhalten; JPEG, GIF und MP4 haben keinen Alphakanal und erhalten stattdessen Weiß.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>Ausgeblendeter Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>Vorschau aktualisieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>FELDDARSTELLUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>Auf alle Felder anwenden</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Farbskala, Deckkraft und Auto-Bereich dieses Felds auf alle aktivierten Felder anwenden (jedes Feld behält sein eigenes Min/Max).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>Vorschau fehlgeschlagen: </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Aktivieren Sie ein Feld im Reiter „Bilder“ für die Vorschau.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>Keine Daten für dieses Feld/Bild.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>BILDBEREICH</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>Alle Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>Von</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1619"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>bis</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1650"/>
        <source>Select Output Folder</source>
        <translation>Ausgabeordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>%1 Dateien exportiert → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>Fehler: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>Wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Rendere %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>Bild %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>%1 Bilder exportiert → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>Es wurde keine Animation geschrieben. Einzelheiten siehe Protokoll.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>Bericht gespeichert → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="18"/>
        <source>Disp U</source>
        <translation>Versch. U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Disp V</source>
        <translation>Versch. V</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>Animation abspielen</translation>
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
        <translation>Nächstes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>Wiedergabegeschwindigkeit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>BILD 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>Animation pausieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>BILD %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Laden Sie zuerst Bilder, bevor Sie eine Region of Interest zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Die drei Punkte sind fast kollinear — wählen Sie Punkte, die über den Kreisrand verteilt sind.</translation>
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
        <translation>Dateiname</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Region</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="190"/>
        <location filename="../../gui/widgets/image_list.py" line="248"/>
        <source>Add</source>
        <translation>Hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="242"/>
        <source>Edit</source>
        <translation>Bearb.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>Offen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>Region of Interest für %n Bilder importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Region of Interest löschen (%1 mit Region)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>Region of Interest löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>%n Bilder löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%1 Dateien für %2 Bilder ausgewählt — Anzahl muss übereinstimmen</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>Startpunkte</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>Platzieren Sie einige Punkte; pyALDIC initialisiert jeden mit einer Einpunkt-NCC und propagiert das Feld entlang der Netz-Nachbarn.

Optimal für:
• Große Verschiebungen zwischen Bildern (&gt; 50 px)
• Diskontinuierliche Felder (Risse, Scherbänder)
• Szenarien, in denen FFT falsche Peaks wählt

Beim Zeichnen oder Bearbeiten einer ROI automatisch pro Region platziert.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>Startpunkte platzieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>Platzierungsmodus auf der Zeichenfläche aktivieren. Linksklick zum Hinzufügen, Rechtsklick zum Entfernen, Esc oder erneuter Klick zum Beenden.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>Auto-Platzieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>Leere Regionen mit dem Knoten mit höchstem NCC-Wert füllen. Vorhandene Startpunkte bleiben erhalten.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>Alle Startpunkte entfernen. Schneller als jeden einzeln per Rechtsklick zu löschen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 Regionen bereit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT (Kreuzkorrelation)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>Normalisierte Kreuzkorrelation auf dem gesamten Gitter. Robust innerhalb des Suchradius; die Suche erweitert sich automatisch bei abgeschnittenen Peaks.

Optimal für:
• Kleine bis mittlere gleichmäßige Bewegungen
• Gut texturierte Speckles
• Keine spezielle Nutzerkonfiguration erforderlich

Aufwand wächst mit dem Suchradius, sehr große Verschiebungen werden langsam.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>Alle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>FFT alle N Bilder ausführen. N = 1 bedeutet FFT in jedem Bild (sicherste, langsamste Option). N &gt; 1 verwendet Warmstart zwischen Resets, um die Fehlerausbreitung auf N Bilder zu begrenzen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>(N=1 = jedes Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>Nur bei Referenzbild-Aktualisierung (nur inkrementell)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>FFT bei jedem Wechsel des Referenzbildes ausführen; Warmstart innerhalb jedes Segments. Typischer Standard für den inkrementellen Modus.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>Die konvergierte Verschiebung des vorherigen Bildes als Anfangsschätzung verwenden. Keine Kreuzkorrelation wird ausgeführt.

Optimal für:
• Sehr kleine Bewegungen zwischen Bildern (wenige Pixel)
• Schnellste Option bei gleichmäßiger Bewegung

Fehler können sich über lange Sequenzen akkumulieren. Bei verrauschten Daten oder größerer Bewegung FFT oder Startpunkte bevorzugen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>Platzieren… (zum Beenden klicken)</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>BILDER</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="185"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>Natürliche Sortierung (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="188"/>
        <source>Sort by embedded numbers: image1, image2, …, image10
Default (unchecked): lexicographic — best for zero-padded names</source>
        <translation>Nach eingebetteten Zahlen sortieren: image1, image2, …, image10
Standard (nicht aktiviert): lexikographisch — ideal für nullgefüllte Namen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>WORKFLOW-TYP</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>ANFANGSSCHÄTZUNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>REGION OF INTEREST</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>PARAMETER</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>ERWEITERT</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>Kraftdaten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>Keine Datei gewählt.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>Datei wählen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>Kraftspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>Nach Zeit</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>Nach Bildnummer</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>Zeilen den Bildern zuordnen:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>Zeitspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>Die Maschinenzeit beim Referenzbild. Startete die Kamera 2 s nach der Maschine, geben Sie 2 ein.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>Versatz:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>Bildspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>Das erste Bild hat die Nummer:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>Anfangsquerschnitt für die technische Spannung F / A0 in MPa. Für nur Kraft bei 0 lassen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>Querschnitt A0:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>Kraftdaten entfernen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>Kamera-Bildrate: %1 fps, aus „Physikalische Einheiten“.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>Legen Sie unter „Physikalische Einheiten“ die Kamera-Bildrate fest, um nach Zeit zuzuordnen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(unbenannt)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>Kein Bild liegt im Aufzeichnungsbereich: Prüfen Sie die Spalten und den Versatz.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>Bilder mit Kraftwert: %1 von %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>Kraftdaten öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV-Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>%1 konnte nicht gelesen werden: %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>Datei</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>Sitzung öffnen…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>Sitzung speichern…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>.aldic-Dateien mit pyALDIC verknüpfen…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>Registriert .aldic, sodass ein Doppelklick auf eine Sitzungsdatei pyALDIC öffnet (nur aktueller Benutzer, keine Administratorrechte nötig).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>Beenden</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>Einstellungen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>Sprache</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>Sprache geändert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>Sprache auf %1 eingestellt. Bitte starten Sie pyALDIC neu, damit alle Elemente die neue Sprache übernehmen.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>Sitzung speichern</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC-Sitzung</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>groß</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>Ergebnisse einbeziehen?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>Die berechneten Ergebnisse in diese Sitzung einbeziehen?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>Mit Ergebnissen (etwa %1 unkomprimiert) können Sie die Sitzung ohne Neuberechnung wieder öffnen. Wählen Sie Nein, um eine kleine reine Konfigurationsdatei zum Teilen zu speichern.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>Sitzung wird gespeichert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>Speichern der Sitzung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>Sitzung öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>Sitzung wird geladen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>Öffnen der Sitzung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>Sitzungsbilder suchen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>Der mit dieser Sitzung gespeicherte Bildordner wurde nicht gefunden:
%1

Die Ergebnisse wurden wiederhergestellt. Wählen Sie den Ordner, der die Bilder jetzt enthält, um den Hintergrund anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>Bildordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>Dateiverknüpfung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>.aldic-Dateien konnten nicht registriert werden: </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>Dateiverknüpfung</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>Fertig. Ein Doppelklick auf eine .aldic-Datei öffnet nun pyALDIC und stellt diese Sitzung wieder her.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>Bild %1 hat keine eigene Region of Interest — für die Berechnung wird die von Bild 1 verwendet. Wechseln Sie zu Bild 1, um sie zu bearbeiten, oder importieren Sie eine Maske, um diesem Bild eine eigene zu geben.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>Keine Region of Interest zum Speichern — laden Sie zuerst Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>Die Region-of-Interest-Maske ist leer.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>Region-of-Interest-Maske speichern</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>PNG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>Maske gespeichert unter %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>Keine Region of Interest zum Invertieren — laden Sie zuerst Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>Bitte zuerst Bilder laden.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>Definieren Sie zuerst eine Region of Interest auf Bild 1.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Maske für Bild %1 importiert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>Stapelimport: %n Maske(n) geladen</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>Netzfarbe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>Klicken, um die Netzlinienfarbe zu wählen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>Linienbreite</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="48"/>
        <source>Subset Size</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past a cap of about 28 characters of the label font, wrapped rather than squeezing the inputs.</extracomment>
        <translation>Subset-Größe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="54"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>IC-GN-Subset-Fenstergröße in Pixeln (ungerade Zahl)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <source>Subset Step</source>
        <translation>Subset-Schritt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="64"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>Knotenabstand in Pixeln (muss eine Zweierpotenz sein)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="72"/>
        <location filename="../../gui/widgets/param_panel.py" line="197"/>
        <location filename="../../gui/widgets/param_panel.py" line="209"/>
        <source>Search Range</source>
        <translation>Suchbereich</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="87"/>
        <source>Refine Inner Boundary</source>
        <translation>Innere Grenze verfeinern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>Netz entlang innerer Maskenränder lokal verfeinern
(Löcher in der Region of Interest). Nützlich für Blasen- oder Porenränder.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="94"/>
        <source>Refine Outer Boundary</source>
        <translation>Äußere Grenze verfeinern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="97"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>Netz entlang des äußeren Rands der Region of Interest lokal verfeinern.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="112"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>Verfeinerungsstärke. Minimale Elementgröße = max(2, subset_step / 2^level). Wird gleichmäßig auf innere, äußere Grenzen UND mit dem Pinsel gemalte Verfeinerungszonen angewendet. Verfügbare Stufen hängen von Subset-Größe und Subset-Schritt ab.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="118"/>
        <source>Refinement Level</source>
        <translation>Verfeinerungsstufe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>Maximale Verschiebung pro Bild, die die FFT-Suche erkennen kann (Pixel).
Deutlich größer als die erwartete Bewegung zwischen Bildern einstellen.
Für große Rotationen im inkrementellen Modus muss dies abdecken:
  Radius × sin(Winkel pro Schritt).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="185"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>Anfängliche Halbbreite (Pixel) der Einpunkt-NCC-Suche an jedem Startpunkt.
Erweitert sich bei abgeschnittenem Peak automatisch um den Faktor 2 pro Wiederholung bis zur halben Bildgröße.
Betrifft nur die Initialisierung der Startpunkte; andere Knoten verwenden F-aware-Propagation (keine knotenweise Suche).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="193"/>
        <location filename="../../gui/widgets/param_panel.py" line="209"/>
        <source>Starting Point Search</source>
        <translation>Startpunkt-Suche</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="249"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>Leicht</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>Mittel</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="251"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Stark</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="252"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Sehr stark</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="253"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>Ultra</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="259"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="281"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>min. Elementgröße = %1 px  (subset_step=%2, level=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>Physikalische Einheiten verwenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>Physikalische Größe eines Bildpixels</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="80"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>Pixel size</source>
        <translation>Pixelgröße</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>Aufnahme-Bildrate (für das Geschwindigkeitsfeld verwendet)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>Bildrate</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>Verschiebung: %1  Geschwindigkeit: %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>Verschiebung: px  Geschwindigkeit: px/Bild</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>Pipeline-Konfiguration wird erstellt…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>Bilder werden geladen…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  %1 Bilder geladen, shape=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  ROI-Maske: %1, %2 Pixel (%3%)</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>Lauf abgebrochen: Definieren Sie pro Bild Regions of Interest für die fehlenden Referenzbilder, oder akzeptieren Sie beim nächsten Lauf die vom 1. Bild geerbte Maske.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n Bilder mit benutzerdefinierten ROI-Masken</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>Ergebnisse empfangen: %n Bilder</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>DIC-Analyse wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>Analyse in %1 s abgeschlossen</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>Analyse wurde vom Benutzer gestoppt.</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>Laden Sie zuerst Bilder und zeichnen Sie dann eine Region of Interest auf Bild 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Akkumulativer Modus&lt;/b&gt; — nur Bild 1 benötigt eine Region of Interest. Alle späteren Bilder werden direkt damit verglichen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;Inkrementell, jedes Bild&lt;/b&gt; — Bild 1 benötigt eine Region of Interest. Sie wird automatisch auf jedes spätere Bild vorwärts übertragen (kein bildweises Zeichnen erforderlich).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;Inkrementell, alle %1 Bilder&lt;/b&gt; — Region of Interest auf folgenden Bildern zeichnen: &lt;b&gt;%2&lt;/b&gt; (insgesamt %3 Referenzbilder).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;Inkrementell, benutzerdefiniert&lt;/b&gt; — keine benutzerdefinierten Referenzbilder festgelegt. Bild 1 wird die einzige Referenz sein; fügen Sie weitere Indizes im Feld „Referenzbilder“ hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;Inkrementell, benutzerdefiniert&lt;/b&gt; — Region of Interest auf folgenden Bildern zeichnen: &lt;b&gt;%1&lt;/b&gt; (insgesamt %2 Referenzbilder).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>Zeichnen Sie eine Region of Interest auf Bild 1.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ Hinzufügen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Region zur Region of Interest hinzufügen (Polygon / Rechteck / Kreis)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>Ausschneiden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Region aus der Region of Interest ausschneiden (Polygon / Rechteck / Kreis)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ Verfeinern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>Zusätzliche Netzverfeinerungszonen mit einem Pinsel malen
(nur auf Bild 1 — Materialpunkte werden automatisch auf spätere Bilder übertragen)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>Der Verfeinerungspinsel ist nur auf Bild 1 verfügbar. Wechseln Sie zu Bild 1, um Verfeinerungszonen zu malen; sie werden automatisch auf spätere Bilder übertragen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>Importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>Maske aus Bilddatei importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>Stapelimport</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>Maskendateien für mehrere Bilder stapelweise importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>Speichern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>Aktuelle Maske als PNG-Datei speichern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>Invertieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>Region-of-Interest-Maske invertieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>Alle Region-of-Interest-Masken löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>Radius</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>Malen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>Radieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Pinsel löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>Polygon</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>Rechteck</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>Kreis</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>Kreis (3 Punkte)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>Import Mask Image</source>
        <translation>Maskenbild importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="356"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>DIC-Analyse ausführen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>Abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>Aktuelle Analyse abbrechen. Bereits berechnete Bilder bleiben erhalten, sodass Sie den Teillauf ansehen oder exportieren können.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>Dehnungsfenster öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>Dehnung in einem separaten Nachbearbeitungsfenster berechnen und visualisieren. Benötigt Verschiebungsergebnisse eines abgeschlossenen Laufs.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>FORTSCHRITT</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>Bereit</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>VERSTRICHEN  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>VERBLEIBEND  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>FELD</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>Anzeigen auf</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Das Feld an den verformten Knotenpositionen zeichnen oder an ihren Positionen im Referenzbild.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>Ausgeblendeter Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="178"/>
        <source>White</source>
        <translation>Weiß</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Black</source>
        <translation>Schwarz</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="183"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Was das Bild ersetzt, wenn es ausgeblendet ist. Beim Export bleibt die Transparenz bei PNG und TIFF erhalten; andere Formate erhalten Weiß.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISIERUNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Deckkraft der Überlagerung (0 = transparent, 100 = deckend)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>PROTOKOLL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Platzieren Sie vor dem Ausführen mindestens einen Startpunkt in jeder roten Region (rot = Startpunkt benötigt).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Bild %2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="150"/>
        <source>DISPLACEMENT</source>
        <translation>VERSCHIEBUNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="161"/>
        <source>STRAIN</source>
        <translation>DEHNUNG</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>Animation abspielen</translation>
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
        <translation>Nächstes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>Wiedergabegeschwindigkeit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>BILD 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>Animation pausieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>BILD %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>Ebenenanpassung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>FEM-Knoten</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Method</source>
        <translation>Methode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="102"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>VSG (Virtual Strain Gauge) ist der Durchmesser in Pixel des kreisförmigen Bereichs um jeden Netzknoten, der zum Anpassen einer lokalen Verschiebungsebene verwendet wird. Die Dehnung ergibt sich aus der Steigung dieser Ebene.

• Größeres VSG → glattere Dehnung, geringere räumliche Auflösung.
• Kleineres VSG → schärfere Dehnung, mehr Rauschen.
• Faustregel: VSG ≥ 2 × Subset-Schritt + 1 (Standard: 41 px).

Nicht verwendet bei Methode = FEM-Knoten (dort bestimmt der Netzabstand die Größe).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>VSG-Größe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>Anzahl der Netzknoten pro Achse innerhalb des kreisförmigen VSG-Fensters bei gleichmäßigem Netz: 2 × floor(VSG-Radius / Knotenabstand) + 1. Die Ebenenanpassung verwendet jeden Knoten innerhalb des Radius; bei verfeinertem Netz variiert die Anzahl lokal.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>Blendet Dehnung mit geringer Konfidenz an ROI-/Loch-Rändern aus, wo das VSG-Fenster die Grenze überschreitet und die lokale Ebenenanpassung einseitig und unzuverlässig wird.

• Koeffizient × VSG-Radius = Breite des beschnittenen Randbereichs.
• 0.00 = jeden Knoten behalten (kein Beschneiden).
• 0.70 = empfohlen (beschneidet, wo der Randfehler stark ansteigt).
• 1.00 = strengste Einstellung (beschneidet jeden Knoten, dessen Fenster den Rand berührt).

Gilt nur bei Methode = Ebenenanpassung.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>Ränder mit geringer Konfidenz beschneiden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Aus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>Leicht (σ = 0,5 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>Mittel (σ = 1 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>Stark (σ = 2 × step) ⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>Gauß-Glättung des Dehnungsfelds nach der Berechnung.
σ ist die Breite des Gauß-Kerns; „step“ = DIC-Knotenabstand.
  Leicht   (0,5 × step): dezent, feine Merkmale bleiben erhalten.
  Mittel   (1 × step):   ausgewogen, empfohlen für verrauschte Daten.
  Stark    (2 × step) ⚠: aggressiv, kann echte Gradienten verwischen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>Dehnungsfeldglättung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>Infinitesimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>Euler</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>Dehnungstyp</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>Beschnitten: %1 Knoten (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>Dehnungsfenster ≈ %1×%2 Knoten</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ VSG-Radius (%1 px) &lt; DIC-Knotenabstand (%2 px); Ebenenanpassung wird fehlschlagen. VSG ≥ %3 px verwenden oder Methode auf FEM-Knoten wechseln.</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Das Feld an den verformten Knotenpositionen zeichnen oder an ihren Positionen im Referenzbild.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>Anzeigen auf</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="82"/>
        <source>White</source>
        <translation>Weiß</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="83"/>
        <source>Black</source>
        <translation>Schwarz</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="87"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Was das Bild ersetzt, wenn es ausgeblendet ist. Beim Export bleibt die Transparenz bei PNG und TIFF erhalten; andere Formate erhalten Weiß.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>Ausgeblendeter Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>Bereich</translation>
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
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="143"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>Beschnittene Ränder füllen (nur Anzeige)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>Interpoliert das randbeschnittene Dehnungsband aus zuverlässigen inneren Knoten neu. Betrifft die Bildschirmansicht und exportierte Bilder/Animationen; exportierte Datendateien behalten den beschnittenen Rand immer als NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>Ränder</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="155"/>
        <source>Strain Post-Processing</source>
        <translation>Dehnungs-Nachbearbeitung</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit</source>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="204"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="214"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="272"/>
        <source>STRAIN PARAMETERS</source>
        <translation>DEHNUNGSPARAMETER</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="291"/>
        <source>Cancel</source>
        <translation>Abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="295"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Laufende Dehnungsberechnung abbrechen. Das vorherige Dehnungsergebnis bleibt erhalten.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Verschiebungs- und Dehnungsergebnisse als NPZ / MAT / CSV / PNG exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="340"/>
        <source>FIELD</source>
        <translation>FELD</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISIERUNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="362"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="371"/>
        <source>LOG</source>
        <translation>PROTOKOLL</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="400"/>
        <source>Strain Field</source>
        <translation>Dehnungsfeld</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="402"/>
        <source>Analysis</source>
        <translation>Analyse</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="516"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Dehnungsberechnung fehlgeschlagen: %1: %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="525"/>
        <location filename="../../gui/strain_window.py" line="587"/>
        <source>Strain computation complete.</source>
        <translation>Dehnungsberechnung abgeschlossen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="536"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Dehnungsfenster: Keine Verschiebungs-Ergebnisse zur Nachbearbeitung.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="567"/>
        <source>Cancelling…</source>
        <translation>Wird abgebrochen…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="604"/>
        <source>Strain computation cancelled.</source>
        <translation>Dehnungsberechnung abgebrochen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="613"/>
        <source>Strain compute failed: %1</source>
        <translation>Dehnungsberechnung fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain Computation Failed</source>
        <translation>Dehnungsberechnung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="659"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Parameter geändert — „Dehnung berechnen“ klicken</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="220"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="282"/>
        <source>Compute Strain</source>
        <translation>Dehnung berechnen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="304"/>
        <source>Export Results</source>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="547"/>
        <source>Starting…</source>
        <translation>Wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="583"/>
        <source>Complete</source>
        <translation>Abgeschlossen</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>Physikalische Einheiten verwenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>Einheit: px/Bild</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="51"/>
        <source>Incremental</source>
        <translation>Inkrementell</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="52"/>
        <source>Accumulative</source>
        <translation>Akkumulativ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>Inkrementell: Jedes Bild wird mit dem vorherigen Referenzbild verglichen.
Geeignet für große kumulierte Verformungen, erforderlich bei großen Rotationen.

Akkumulativ: Jedes Bild wird mit Bild 1 verglichen.
Nur für kleine, monotone Verformungen genau.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>Tracking-Modus</translation>
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
        <translation>Local DIC: Unabhängiges Subset-Matching (IC-GN). Schnell,
erhält scharfe lokale Merkmale. Optimal für kleine
Verformungen oder hochwertige Bilder.

AL-DIC: Erweiterte Lagrange-Methode mit globaler FEM-
Regularisierung. Erzwingt Verschiebungskompatibilität
zwischen Subsets. Optimal für große Verformungen,
verrauschte Bilder oder hohe Dehnungsgenauigkeit.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>Löser</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>Jedes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>Alle N Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>Benutzerdefiniert</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>Zeitpunkt der Referenzaktualisierung beim inkrementellen Tracking.
Jedes Bild: Referenz bei jedem Bild zurücksetzen (kleinste Schrittverschiebung,
am robustesten für große Verformungen).
Alle N Bilder: alle N Bilder zurücksetzen (Balance zwischen Geschwindigkeit und Robustheit).
Benutzerdefiniert: vom Benutzer festgelegte Liste der Referenzbild-Indizes.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>Referenzaktualisierung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>Referenz alle N Bilder aktualisieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>Intervall</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>Komma-getrennte Bildindizes als Referenzbilder (0-basiert)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>Referenzbilder</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>Bildordner hier ablegen
oder Durchsuchen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>Bildordner auswählen</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>Vorschau</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>(kein Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>Nur Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>Bild + Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>Nur Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>Ansicht:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>Alpha:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Blau</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rot</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Grün</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Gelb</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>Maskenfarbe:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>Keine Maske zugewiesen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>Bild %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>Bild konnte nicht geladen werden</translation>
    </message>
</context>
</TS>
