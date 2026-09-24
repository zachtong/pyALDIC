<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="es" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>Iteraciones AL-DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>Número de ciclos de refinamiento global del solucionador AL-DIC.
1 = pasada única (más rápido), 3 = predeterminado,
5+ = rendimientos decrecientes en la mayoría de los casos.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>Solo afecta al solucionador AL-DIC. Local DIC lo ignora.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>Expandir automáticamente la búsqueda FFT cuando los picos se recortan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>Cuando el pico NCC alcanza el borde de la región de búsqueda, reintenta automáticamente con una región mayor (hasta la mitad del tamaño de la imagen, 6 reintentos con crecimiento ×2).

Solo relevante para el modo de estimación inicial FFT.</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="70"/>
        <source>crack</source>
        <translation>grieta</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="72"/>
        <source>too few valid points</source>
        <translation>muy pocos puntos válidos</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="74"/>
        <source>unreliable (strain edge trim)</source>
        <translation>no fiable (recorte de bordes de la deformación)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>gauge endpoint lost</source>
        <translation>extremo del calibre perdido</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="79"/>
        <source>not computed</source>
        <translation>no calculado</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>no data</source>
        <translation>sin datos</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>Punto</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>Línea</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>Rectángulo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>Círculo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>Polígono</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="106"/>
        <source>Click once to place a point probe.</source>
        <translation>Haga clic una vez para colocar una sonda puntual.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click twice: opposite corners.</source>
        <translation>Haga clic dos veces: esquinas opuestas.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>Haga clic dos veces: centro y luego borde.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="116"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>Haga clic en cada vértice y doble clic para cerrar.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="87"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>Mostrar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>Nombre</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>Tipo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>Color</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>Nota</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="98"/>
        <source>Colour…</source>
        <translation>Color…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="99"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>Eliminar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Clear All</source>
        <translation>Borrar todo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Statistic:</source>
        <translation>Estadístico:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>Media</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>Mediana</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="63"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>Máximo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="64"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>Mínimo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="65"/>
        <source>Standard deviation</source>
        <translation>Desviación típica</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="66"/>
        <source>Valid fraction</source>
        <translation>Fracción válida</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="50"/>
        <source>Crack opening</source>
        <translation>Apertura de grieta</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="487"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>La deformación aún no se ha calculado. Calcúlela en la pestaña «Campo de deformación» o represente un desplazamiento.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="531"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>Las magnitudes de calibre requieren una sonda de línea.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="528"/>
        <source>No visible probe can show this quantity.</source>
        <translation>Ninguna sonda visible puede mostrar esta magnitud.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>no valid data: %1</source>
        <translation>sin datos válidos: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>crack from frame %1</source>
        <translation>grieta desde el fotograma %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="178"/>
        <source>endpoint lost from frame %1</source>
        <translation>extremo perdido desde el fotograma %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="181"/>
        <source>gaps: too few valid points</source>
        <translation>huecos: muy pocos puntos válidos</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="183"/>
        <source>gaps: unreliable strain</source>
        <translation>huecos: deformación no fiable</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="189"/>
        <source>not plotted: gauges need a line</source>
        <translation>no representada: los calibres necesitan una línea</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="192"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>no representada: un solo punto no tiene dispersión ni cobertura</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="194"/>
        <source>not plotted</source>
        <translation>no representada</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="508"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>Sonda «%1» añadida.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="594"/>
        <source>Clear All Probes</source>
        <translation>Borrar todas las sondas</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="595"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>¿Eliminar todas las sondas? Esta acción no se puede deshacer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="108"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>Punto</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="109"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>Línea</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Región</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="478"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>Ejecute un análisis DIC para representar las sondas.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="96"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>Extensómetro virtual</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="98"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>Calibre de grieta</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>Haga clic dos veces: inicio y fin. Una línea es también un extensómetro virtual y un calibre de apertura de grieta.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>Haga clic en los dos puntos de medida. El gráfico muestra entonces la deformación entre ellos.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="122"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>Haga clic en un punto a cada lado de la grieta. El gráfico muestra entonces cuánto se abre.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="122"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>Ajustar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="129"/>
        <source>Fit image to viewport</source>
        <translation>Ajustar la imagen a la vista</translation>
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
        <translation>Zoom al 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="134"/>
        <source>Zoom in</source>
        <translation>Acercar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom out</source>
        <translation>Alejar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Show field</source>
        <comment>Analysis canvas: colour the image by the field</comment>
        <translation>Mostrar campo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Colour the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>Colorea la imagen de referencia con el campo representado en el fotograma actual. Para una lectura de calibre, se muestra el campo de la pestaña «Campo de deformación».</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="179"/>
        <source>Plot:</source>
        <translation>Representar:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="182"/>
        <source>X axis:</source>
        <translation>Eje X:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Strain as:</source>
        <translation>Deformación en:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Min. valid fraction:</source>
        <translation>Fracción válida mín.:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="186"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>Un fotograma queda en blanco cuando menos de esta fracción de los puntos de una línea o región es fiable. Evita una curva que sigue suave mientras su muestra se reduce.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="193"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>Evolución temporal</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>La lectura de cada sonda en cada fotograma.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="198"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>A lo largo de la línea</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>The field along the selected line at the current frame, over the other frames in grey.</source>
        <translation>El campo a lo largo de la línea seleccionada en el fotograma actual, sobre los demás fotogramas en gris.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="205"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>Quimograma</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>The field along the selected line at every frame: distance against frame, value as colour.</source>
        <translation>El campo a lo largo de la línea seleccionada en cada fotograma: distancia frente a fotograma, valor como color.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="212"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>Tensión–deformación</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>Tensión (o carga, sin A0) frente a la magnitud representada, una curva por sonda: tensión-deformación con un extensómetro, carga-apertura con un calibre de grieta. Requiere datos de carga.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="225"/>
        <source>Other frames</source>
        <translation>Otros fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="226"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>Dibujar tenuemente los perfiles de los demás fotogramas detrás del actual (como máximo doce, espaciados uniformemente).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="231"/>
        <source>Line data (CSV)…</source>
        <translation>Datos de línea (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="232"/>
        <source>Export</source>
        <translation>Exportar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Load data…</source>
        <translation>Datos de carga…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>Importar el registro de carga de una máquina de ensayo (CSV) para representar frente a la carga o la tensión y trazar curvas tensión-deformación.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="239"/>
        <source>Probe data (CSV)…</source>
        <translation>Datos de las sondas (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Chart image…</source>
        <translation>Imagen del gráfico…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Copy chart</source>
        <translation>Copiar el gráfico</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy plotted data</source>
        <translation>Copiar los datos representados</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="557"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>Importe el registro de carga de la máquina de ensayo con «Datos de carga…» para trazar curvas tensión-deformación.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="28"/>
        <source>Displacement U</source>
        <translation>Desplazamiento U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="29"/>
        <source>Displacement V</source>
        <translation>Desplazamiento V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="30"/>
        <source>Displacement magnitude</source>
        <translation>Magnitud del desplazamiento</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Extensometer strain</source>
        <translation>Deformación del extensómetro</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Extensometer true strain</source>
        <translation>Deformación verdadera del extensómetro</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="49"/>
        <source>Elongation ΔL</source>
        <translation>Alargamiento ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="51"/>
        <source>Crack sliding</source>
        <translation>Deslizamiento de grieta</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="52"/>
        <source>Crack opening magnitude</source>
        <translation>Magnitud de la apertura de grieta</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="277"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="432"/>
        <source>Frame</source>
        <translation>Fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="427"/>
        <source>Time (s)</source>
        <translation>Tiempo (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="572"/>
        <source>Load (N)</source>
        <translation>Carga (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="284"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="569"/>
        <source>Stress (MPa)</source>
        <translation>Tensión (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="293"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>proporción</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="417"/>
        <source>Could not draw the field: %1</source>
        <translation>No se pudo dibujar el campo: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="134"/>
        <source>Esc cancels placement</source>
        <translation>Esc cancela la colocación</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="142"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>Arrastre para mover la sonda, o un tirador para cambiar su forma. Supr la elimina; F2 le cambia el nombre.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="483"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>Coloque una sonda en la imagen de referencia para empezar.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="205"/>
        <source>Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.</source>
        <translation>No hay datos válidos a lo largo de %1: su deformación se recorta como poco fiable cerca de un borde o un agujero. Represente un desplazamiento, o recorte menos en la pestaña «Campo de deformación».</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="211"/>
        <source>Nothing valid along %1: a crack has consumed the material under it.</source>
        <translation>No hay datos válidos a lo largo de %1: una grieta ha consumido el material que hay debajo.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="216"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>No hay datos válidos a lo largo de %1: está fuera del área medida.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="608"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>Una vista de línea muestra un campo. Elija un campo para representar.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="612"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>Coloque o seleccione una sonda de línea para ver el campo a lo largo de ella.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="630"/>
        <source>Distance along %1 (%2)</source>
        <translation>Distancia a lo largo de %1 (%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="639"/>
        <source>%1, frame %2</source>
        <translation>%1, fotograma %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="157"/>
        <source>not plotted: off the measured area</source>
        <translation>no representada: fuera del área medida</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="160"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>no representada: un extremo del calibre está fuera del área medida</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="662"/>
        <source>Export Probe Data</source>
        <translation>Exportar datos de sonda</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <source>CSV Files</source>
        <translation>Archivos CSV</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="439"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>Los datos de carga no se pueden asociar a los fotogramas: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <location filename="../../gui/panels/analysis/tab.py" line="722"/>
        <source>All Files</source>
        <translation>Todos los archivos</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="691"/>
        <source>Probe export failed: %1</source>
        <translation>Error al exportar la sonda: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="693"/>
        <source>Probe data written to %1</source>
        <translation>Datos de sonda escritos en %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Export Line Data</source>
        <translation>Exportar datos de línea</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="712"/>
        <source>Line export failed: %1</source>
        <translation>Error al exportar la línea: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="714"/>
        <source>Line data written to %1</source>
        <translation>Datos de línea escritos en %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>SVG Images</source>
        <translation>Imágenes SVG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="731"/>
        <source>Export Chart</source>
        <translation>Exportar gráfico</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart copied to the clipboard.</source>
        <translation>Gráfico copiado al portapapeles.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="758"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>Datos representados copiados al portapapeles.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="719"/>
        <source>PNG Images</source>
        <translation>Imágenes PNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="721"/>
        <source>PDF Documents</source>
        <translation>Documentos PDF</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="742"/>
        <source>Chart export failed: %1</source>
        <translation>Error al exportar el gráfico: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="744"/>
        <source>Chart written to %1</source>
        <translation>Gráfico escrito en %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="983"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>Región de interés importada para %n fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>Ejecute primero el DIC — no hay resultados de desplazamiento para posprocesar.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1094"/>
        <source>pyALDIC has hit an error</source>
        <translation>pyALDIC ha encontrado un error</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1095"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>Se ha producido un error inesperado. Es posible que la aplicación no funcione correctamente a partir de ahora; se recomienda guardar la sesión y reiniciar.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1105"/>
        <source>Details were written to %1</source>
        <translation>Los detalles se han escrito en %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1209"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>Preparando los núcleos de cálculo en segundo plano. El primer análisis tras una instalación nueva tarda más que los siguientes.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1223"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>Núcleos de cálculo listos (%1 s).</translation>
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
        <translation>Ajustar el rango de colores al rango de datos de cada fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>Fijo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>Mantener los límites Mín/Máx manuales en todos los fotogramas</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Importar por lotes máscaras de región de interés</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>Carpeta de máscaras:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>(ninguna)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>Examinar…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>Máscaras disponibles</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>Coincidencia automática por nombre</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Asociar archivos de máscara a fotogramas según el número del nombre de archivo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>Asignar secuencialmente</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Asignar máscaras a los fotogramas en orden desde el fotograma 0</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>Asignaciones de fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>Fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>Imagen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>Máscara</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>Asignar selección -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Emparejar las máscaras seleccionadas con los fotogramas seleccionados</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>Borrar todo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>Seleccionar carpeta de máscaras</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>No se pudo leer el archivo de máscara.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Forma no coincide: %1×%2 (se esperaba %3×%4)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>%n máscara(s) tienen tamaños no coincidentes y están deshabilitadas.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>Asignación no válida</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Un fotograma solo puede tener una máscara. Seleccione exactamente una máscara, o seleccione varios fotogramas para asignar una máscara a muchos.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>Ajustar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>Ajustar la imagen a la vista</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoom al 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1226"/>
        <source>Zoom in</source>
        <translation>Acercar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1233"/>
        <source>Zoom out</source>
        <translation>Alejar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>Mostrar cuadrícula</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>Mostrar/ocultar la cuadrícula de la malla</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>Mostrar subconjunto</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>Mostrar ventana del subconjunto al pasar el cursor (requiere cuadrícula)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>Colocando puntos de inicio</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Mode</source>
        <translation>Modo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Solver</source>
        <translation>Solucionador</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="43"/>
        <source>Init</source>
        <translation>Inicial</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>Acumulativo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="94"/>
        <source>Incremental</source>
        <translation>Incremental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="101"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM (%1 iter.)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="110"/>
        <source>Starting Points</source>
        <translation>Puntos de inicio</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>Fotograma anterior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>FFT cada fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>FFT cada %1 fotogramas</translation>
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
        <translation>Rango</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>Mín</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>Máx</translation>
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
        <translation>Opacidad</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Opacidad del campo (0 = transparente, 1 = completamente opaco)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>Todos</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>Ninguno</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>Exportar resultados</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>CARPETA DE SALIDA</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>Seleccionar carpeta de salida…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>Examinar…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>Abrir carpeta</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNIDADES FÍSICAS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>Activar unidades físicas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Escalar los valores de desplazamiento por el tamaño del píxel y mostrar unidades físicas en las etiquetas de la barra de color. La deformación es adimensional y no se ve afectada.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="695"/>
        <source>/ pixel</source>
        <translation>/ píxel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="697"/>
        <source>Pixel size</source>
        <translation>Tamaño del píxel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="712"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="714"/>
        <source>Frame rate</source>
        <translation>Fotogramas por segundo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>Datos</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>Imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>Animación</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>Informe</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>Vista previa y barra de color</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>FORMATO</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>Archivo NumPy (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV (por fotograma)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ: un archivo por fotograma (predeterminado: un único archivo combinado)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="775"/>
        <source>DISPLACEMENT</source>
        <translation>DESPLAZAMIENTO</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Select:</source>
        <translation>Seleccionar:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>DEFORMACIÓN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>Ejecute primero «Calcular deformación».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ El archivo de parámetros (JSON) siempre se exporta</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>Exportar datos</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>Exportar</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>Campo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>Mapa de colores</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="860"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1009"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1367"/>
        <source>Min</source>
        <translation>Mín</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="861"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1010"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1374"/>
        <source>Max</source>
        <translation>Máx</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="880"/>
        <source>IMAGE SETTINGS</source>
        <translation>AJUSTES DE IMAGEN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="890"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1039"/>
        <source>Format</source>
        <translation>Formato</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="898"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1047"/>
        <source>Full resolution</source>
        <translation>Resolución completa</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>Limita el borde largo de la imagen exportada (el mayor de ancho/alto; se mantiene la relación de aspecto).
El detalle del campo está limitado por la malla, por lo que un límite menor es casi sin pérdida,
pero mucho más pequeño y rápido de codificar. Menor = más rápido. «Resolución completa» mantiene el tamaño nativo.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>Resolución (borde largo)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>Fotograma deformado</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>Fotograma de referencia</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>Fotograma deformado: el campo se dibuja en las posiciones desplazadas de los nodos (referencia + desplazamiento), sobre la foto propia de cada fotograma.
Fotograma de referencia: se dibuja en las posiciones originales de los nodos, sobre el primer fotograma.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>Mostrar imagen de fondo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>Desmarque para exportar solo el campo, sin imagen de moteado detrás. El relleno se elige en la pestaña Preview &amp; Colorbar.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>Limita el borde largo de la animación (el mayor de ancho/alto).
Menor = más rápido y mucho más pequeño. Muy recomendable para GIF, cuyo tamaño se dispara a resolución nativa.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>Calidad JPEG (mayor = archivo más grande). Se ignora para PNG/TIFF.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>Calidad JPEG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="927"/>
        <source>DPI</source>
        <translation>PPP</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="929"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1080"/>
        <source>Include colorbar</source>
        <translation>Incluir barra de color</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Añade una barra de color vertical a la derecha de cada imagen.
Las etiquetas se actualizan por fotograma cuando el rango auto está activo.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>Representar como</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>Cancelar exportación</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>Exportar imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>AJUSTES DE ANIMACIÓN</translation>
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
        <translation>Exporta uno de cada N fotogramas (1 = todos). Mayor = más rápido y pequeño,
pero se ve más entrecortado. La duración se conserva (los FPS de arriba son la tasa antes de diezmar).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>Paso de fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>Añade una barra de color vertical a la derecha de cada fotograma.
Las etiquetas se actualizan por fotograma cuando el rango auto está activo.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>Desmarque para exportar solo el campo, sin imagen de moteado detrás. GIF y MP4 no pueden almacenar transparencia, por lo que un relleno transparente se escribe como blanco.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>Exportar animación</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>CONTENIDO</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>Tabla resumen de parámetros</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Estadísticas de campo (mín/máx/media/desv.típ. por fotograma)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>Imágenes de campo de muestra</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1164"/>
        <source>Sample every</source>
        <translation>Muestrear cada</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1170"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>CAMPOS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>Desplazamiento:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>Deformación:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Formato: HTML (autocontenido, se puede ver en cualquier navegador)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>Generar informe</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>Abre esta pestaña para generar una vista previa.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>Fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>ESTILO DE BARRA DE COLOR</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Right</source>
        <translation>Derecha</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Left</source>
        <translation>Izquierda</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Top</source>
        <translation>Arriba</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Bottom</source>
        <translation>Abajo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1280"/>
        <source>Position</source>
        <translation>Posición</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1286"/>
        <source>Font size</source>
        <translation>Tamaño de fuente</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>Fuente</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>Grosor de la barra</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1326"/>
        <source>Black</source>
        <translation>Negro</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1325"/>
        <source>White</source>
        <translation>Blanco</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1306"/>
        <source>Background</source>
        <translation>Fondo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1313"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>Añade un borde en blanco alrededor del contenido exportado, como fracción del borde largo (0 = ninguna).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>Margen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>Color del margen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <source>Transparent</source>
        <translation>Transparente</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1333"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.</source>
        <translation>Relleno utilizado donde habría estado la imagen de fondo, cuando «Mostrar imagen de fondo» está desactivado.
La transparencia se conserva para PNG y TIFF; JPEG, GIF y MP4 no tienen canal alfa y reciben blanco.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>Fondo oculto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>Actualizar vista previa</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>APARIENCIA DEL CAMPO</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>Rango</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>Aplicar a todos los campos</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Aplica el colormap, la opacidad y el rango automático de este campo a todos los campos activados (cada campo conserva su propio mín/máx).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>Error en la vista previa: </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Active un campo en la pestaña Images para la vista previa.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>No hay datos para este campo/fotograma.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>RANGO DE FOTOGRAMAS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>Todos los fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>Desde</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1619"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>a</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1650"/>
        <source>Select Output Folder</source>
        <translation>Seleccionar carpeta de salida</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>Exportados %1 archivos → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>Error: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>Iniciando…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Renderizando %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>Fotograma %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>%1 imágenes exportadas → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>No se ha escrito ninguna animación. Consulte el registro para más detalles.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>Informe guardado → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="18"/>
        <source>Disp U</source>
        <translation>Despl. U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Disp V</source>
        <translation>Despl. V</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>Fotograma anterior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>Reproducir animación</translation>
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
        <translation>Fotograma siguiente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>Velocidad de reproducción</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>FOTOGRAMA 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>Pausar animación</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>FOTOGRAMA %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Cargue primero las imágenes antes de dibujar una región de interés.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Los tres puntos son casi colineales — elija puntos repartidos por el borde del círculo.</translation>
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
        <translation>Nombre de archivo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Región</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="190"/>
        <location filename="../../gui/widgets/image_list.py" line="248"/>
        <source>Add</source>
        <translation>Añadir</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="242"/>
        <source>Edit</source>
        <translation>Editar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>Falta</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>Importar región de interés para %n fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Borrar región de interés (%1 con región)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>Borrar región de interés</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>Eliminar %n imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>Imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>Todos los archivos</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>Seleccionados %1 archivos para %2 fotogramas — las cantidades deben coincidir</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>Puntos de inicio</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>Coloque algunos puntos; pyALDIC inicializa cada uno con una NCC puntual y propaga el campo a lo largo de los vecinos de la malla.

Ideal para:
• Grandes desplazamientos entre fotogramas (&gt; 50 px)
• Campos discontinuos (grietas, bandas de cortante)
• Escenarios donde la FFT elige picos incorrectos

Se colocan automáticamente por región al dibujar o editar una ROI.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>Colocar puntos de inicio</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>Entrar en modo de colocación en el lienzo. Clic izquierdo para añadir, clic derecho para eliminar, Esc o nuevo clic para salir.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>Colocación automática</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>Rellenar las regiones vacías con el nodo de mayor NCC en cada una. Se conservan los puntos de inicio existentes.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>Limpiar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>Eliminar todos los puntos de inicio. Más rápido que hacer clic derecho en cada uno.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 regiones listas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT (correlación cruzada)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>Correlación cruzada normalizada en toda la cuadrícula. Robusta dentro del radio de búsqueda; la búsqueda se expande automáticamente cuando los picos se recortan.

Ideal para:
• Movimientos suaves pequeños o moderados
• Moteado bien texturado
• No se requiere configuración especial del usuario

El coste crece con el radio de búsqueda, por lo que desplazamientos muy grandes se vuelven lentos.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>Cada</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>Ejecutar FFT cada N fotogramas. N = 1 significa FFT en cada fotograma (más seguro, más lento). N &gt; 1 usa arranque en caliente entre reinicios para limitar la propagación de errores a N fotogramas.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>(N=1 = cada fotograma)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>Solo cuando se actualiza el fotograma de referencia (solo incremental)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>Ejecutar FFT siempre que cambie el fotograma de referencia; arranque en caliente dentro de cada segmento. Valor predeterminado típico para el modo incremental.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>Fotograma anterior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>Usar el desplazamiento convergido del fotograma anterior como estimación inicial. No se ejecuta correlación cruzada.

Ideal para:
• Movimientos entre fotogramas muy pequeños (unos pocos píxeles)
• La opción más rápida cuando el movimiento es suave

Los errores pueden acumularse en secuencias largas. Prefiera FFT o puntos de inicio con datos ruidosos o cuando el movimiento sea mayor.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>Colocando… (clic para salir)</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>IMÁGENES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="185"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>Orden natural (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="188"/>
        <source>Sort by embedded numbers: image1, image2, …, image10
Default (unchecked): lexicographic — best for zero-padded names</source>
        <translation>Ordenar por números incrustados: image1, image2, …, image10
Predeterminado (desmarcado): lexicográfico — ideal para nombres con ceros a la izquierda</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>TIPO DE FLUJO</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>ESTIMACIÓN INICIAL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>REGIÓN DE INTERÉS</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>PARÁMETROS</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>AVANZADO</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>Datos de carga</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>Ningún archivo elegido.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>Elegir archivo…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>Columna de carga:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>Por tiempo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>Por número de fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>Asociar filas a fotogramas:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>Columna de tiempo:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>El tiempo de la máquina en la imagen de referencia. Si la cámara empezó 2 s después que la máquina, introduzca 2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>Desfase:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>Columna de fotograma:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>Número de la primera imagen:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>Sección inicial, para la tensión ingenieril F / A0 en MPa. Déjela en 0 para usar solo la carga.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>Sección A0:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>Quitar los datos de carga</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>Fotogramas por segundo de la cámara: %1 fps, de «Unidades físicas».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>Defina los fotogramas por segundo de la cámara en «Unidades físicas» para asociar por tiempo.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(sin nombre)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>Ningún fotograma cae dentro del registro: compruebe las columnas y el desfase.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>Fotogramas con carga: %1 de %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>Abrir datos de carga</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>Archivos CSV</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>Todos los archivos</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>No se pudo leer %1: %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>Archivo</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>Abrir sesión…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>Guardar sesión…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>Asociar archivos .aldic con pyALDIC…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>Registra .aldic para que hacer doble clic en un archivo de sesión abra pyALDIC (solo el usuario actual, sin permisos de administrador).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>Salir</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>Configuración</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>Idioma</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>Idioma cambiado</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>Idioma establecido en %1. Reinicie pyALDIC para que todos los elementos adopten el nuevo idioma.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>Guardar sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>Sesión de pyALDIC</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>Todos los archivos</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>grande</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>¿Incluir resultados?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>¿Incluir los resultados calculados en esta sesión?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>Incluir los resultados (unos %1 sin comprimir) permite reabrir la sesión sin recalcular. Elija No para guardar un pequeño archivo solo de configuración para compartir.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>Guardando sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>Error al guardar la sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>Abrir sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>Cargando sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>Error al abrir la sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>Localizar imágenes de la sesión</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>No se encontró la carpeta de imágenes guardada con esta sesión:
%1

Los resultados se restauraron. Para mostrar las imágenes de fondo, seleccione la carpeta que ahora las contiene.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>Seleccionar carpeta de imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>Error al asociar archivos</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>No se pudieron registrar los archivos .aldic: </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>Asociación de archivos</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>Listo. Ahora, hacer doble clic en un archivo .aldic abrirá pyALDIC y restaurará esa sesión.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>El fotograma %1 no tiene una región de interés propia — se usa la del fotograma 1 para el cálculo. Cambie al fotograma 1 para editarla, o importe una máscara para dar a este fotograma la suya.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>No hay región de interés que guardar — cargue primero las imágenes.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>La máscara de la región de interés está vacía.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>Guardar máscara de región de interés</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>Imágenes PNG</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>Máscara guardada en %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>No hay región de interés que invertir — cargue primero las imágenes.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>Cargue primero las imágenes.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>Defina primero una región de interés en el fotograma 1.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Máscara importada para el fotograma %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>Importación por lotes: %n máscara(s) cargada(s)</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>Color de malla</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>Haga clic para elegir el color de las líneas de la malla</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>Grosor de línea</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="37"/>
        <source>Subset Size</source>
        <translation>Tamaño del subconjunto</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="43"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>Tamaño de la ventana del subconjunto IC-GN en píxeles (número impar)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="50"/>
        <source>Subset Step</source>
        <translation>Paso del subconjunto</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="53"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>Espaciado de nodos en píxeles (debe ser potencia de 2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Search Range</source>
        <translation>Rango de búsqueda</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Refine Inner Boundary</source>
        <translation>Refinar borde interior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="79"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>Refinar localmente la malla a lo largo de los bordes internos de la máscara
(agujeros dentro de la región de interés). Útil para bordes de burbujas o huecos.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="83"/>
        <source>Refine Outer Boundary</source>
        <translation>Refinar borde exterior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>Refinar localmente la malla a lo largo del borde exterior de la región de interés.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>Intensidad del refinamiento. Tamaño mínimo de elemento = max(2, subset_step / 2^nivel). Se aplica uniformemente a bordes interiores, exteriores Y zonas pintadas con el pincel. Los niveles disponibles dependen del tamaño y el paso del subconjunto.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>Nivel de refinamiento</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>Desplazamiento máximo por fotograma que puede detectar la búsqueda FFT (píxeles).
Configúrelo claramente mayor que el movimiento esperado entre fotogramas.
Para grandes rotaciones en modo incremental, debe cubrir:
  radio × sin(ángulo por paso).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>Semianchura inicial (píxeles) de la búsqueda NCC puntual en cada punto de inicio.
Se expande automáticamente 2× por reintento si el pico se recorta, hasta la mitad del tamaño de la imagen.
Solo afecta a la inicialización de los puntos de inicio; los demás nodos usan propagación F-aware (sin búsqueda por nodo).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Initial Seed Search</source>
        <translation>Búsqueda inicial de semilla</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="218"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>Ligero</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="219"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>Medio</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="220"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Fuerte</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="221"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>Muy fuerte</translation>
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
        <translation>tamaño mín. de elemento = %1 px  (subset_step=%2, nivel=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>Usar unidades físicas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>Tamaño físico de un píxel de la imagen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="80"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>Pixel size</source>
        <translation>Tamaño de píxel</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>Frecuencia de adquisición (usada para el campo de velocidad)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>Velocidad de fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>Despl.: %1  Velocidad: %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>Despl.: px  Velocidad: px/fot.</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>Construyendo configuración del flujo…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>Cargando imágenes…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  %1 imágenes cargadas, forma=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  Máscara ROI: %1, %2 píxeles (%3%)</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>Ejecución cancelada: defina regiones de interés por fotograma para los fotogramas de referencia que faltan, o acepte la máscara heredada del fotograma 1 en la próxima ejecución.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n fotogramas con máscaras ROI personalizadas</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>Resultados recibidos: %n fotogramas</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>Iniciando análisis DIC…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>Análisis completado en %1 s</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>Análisis detenido por el usuario.</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>Cargue primero las imágenes y luego dibuje una región de interés en el fotograma 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Modo acumulativo&lt;/b&gt; — solo el fotograma 1 requiere una región de interés. Todos los fotogramas posteriores se comparan directamente con ella.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;Incremental, cada fotograma&lt;/b&gt; — el fotograma 1 requiere una región de interés. Se propaga automáticamente hacia cada fotograma posterior (no es necesario dibujar por fotograma).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;Incremental, cada %1 fotogramas&lt;/b&gt; — dibuje una región de interés en los fotogramas: &lt;b&gt;%2&lt;/b&gt; (%3 fotogramas de referencia en total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;Incremental, personalizado&lt;/b&gt; — no hay fotogramas de referencia personalizados definidos. El fotograma 1 será la única referencia; añada más índices en el campo «Fotogramas de referencia».</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;Incremental, personalizado&lt;/b&gt; — dibuje una región de interés en los fotogramas: &lt;b&gt;%1&lt;/b&gt; (%2 fotogramas de referencia en total).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>Dibuje una región de interés en el fotograma 1.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ Añadir</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Añadir región a la región de interés (Polígono / Rectángulo / Círculo)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>Recortar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Recortar región de la región de interés (Polígono / Rectángulo / Círculo)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ Refinar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>Pintar zonas adicionales de refinamiento de malla con un pincel
(solo en el fotograma 1 — los puntos materiales se propagan automáticamente a los fotogramas posteriores)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>El pincel de refinamiento solo está disponible en el fotograma 1. Cambie al fotograma 1 para pintar zonas de refinamiento; se propagan automáticamente a los fotogramas posteriores.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>Importar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>Importar máscara desde archivo de imagen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>Importación por lotes</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>Importar por lotes archivos de máscara para varios fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>Guardar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>Guardar la máscara actual en archivo PNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>Invertir</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>Invertir la máscara de la región de interés</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>Limpiar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>Limpiar todas las máscaras de región de interés</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>Radio</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>Pintar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>Borrar</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Limpiar pincel</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="256"/>
        <source>Circle (3-point)</source>
        <translation>Círculo (3 puntos)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="350"/>
        <source>Import Mask Image</source>
        <translation>Importar imagen de máscara</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Images</source>
        <translation>Imágenes</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>All Files</source>
        <translation>Todos los archivos</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>Ejecutar análisis DIC</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>Cancelar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>Cancelar el análisis actual. Los fotogramas ya calculados se conservan, por lo que puede revisar o exportar la ejecución parcial.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>Exportar resultados</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>Abrir ventana de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>Calcular y visualizar la deformación en una ventana de post-procesado separada. Requiere resultados de desplazamiento de una ejecución completada.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>PROGRESO</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>Listo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>TRANSCURRIDO  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>RESTANTE  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>CAMPO</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>Mostrar en</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>Fotograma deformado</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>Fotograma de referencia</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Dibujar el campo en las posiciones deformadas de los nodos, o en sus posiciones en el fotograma de referencia.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>Mostrar imagen de fondo</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Desmarque para mostrar solo el campo, sin imagen de moteado detrás.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>Fondo oculto</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="178"/>
        <source>White</source>
        <translation>Blanco</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Black</source>
        <translation>Negro</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Transparent</source>
        <translation>Transparente</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="183"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Lo que reemplaza a la imagen cuando está oculta. Al exportar, la transparencia se conserva para PNG y TIFF; los demás formatos reciben blanco.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>VISUALIZACIÓN</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>Paleta de colores</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>Opacidad</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Opacidad de la superposición (0 = transparente, 100 = opaco)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNIDADES FÍSICAS</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>REGISTRO</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>Limpiar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Coloque al menos un punto de inicio en cada región roja antes de ejecutar (rojo = requiere punto de inicio).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Fotograma %2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="150"/>
        <source>DISPLACEMENT</source>
        <translation>DESPLAZAMIENTO</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="161"/>
        <source>STRAIN</source>
        <translation>DEFORMACIÓN</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>Fotograma anterior</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>Reproducir animación</translation>
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
        <translation>Fotograma siguiente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>Velocidad de reproducción</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>FOTOGRAMA 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>Pausar animación</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>FOTOGRAMA %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>Ajuste de plano</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>FEM nodal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Method</source>
        <translation>Método</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="102"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>El tamaño VSG (Virtual Strain Gauge, galga de deformación virtual) es el diámetro, en píxeles, de la región circular alrededor de cada nodo de malla utilizada para ajustar un plano de desplazamiento local. La deformación se toma como la pendiente de dicho plano.

• VSG más grande → deformación más suave, menor resolución espacial.
• VSG más pequeño → deformación más nítida, pero con más ruido.
• Regla práctica: VSG ≥ 2 × paso del subset + 1 (predeterminado: 41 px).

No se usa con Method = FEM nodal (allí el espaciado de la malla establece el tamaño).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>Tamaño VSG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>Número de nodos de malla por eje dentro de la ventana VSG circular en una malla uniforme: 2 × floor(radio VSG / espaciado de nodos) + 1. El ajuste de plano usa todos los nodos dentro del radio; en una malla refinada, ese número varía localmente.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>Oculta la deformación de baja confianza en los bordes de la ROI / huecos, donde la ventana VSG cruza el límite y el ajuste de plano local se vuelve unilateral y poco fiable.

• Coeficiente × radio VSG = ancho de la banda de borde recortada.
• 0.00 = conservar todos los nodos (sin recorte).
• 0.70 = recomendado (recorta donde el error de borde aumenta bruscamente).
• 1.00 = más estricto (recorta cualquier nodo cuya ventana toque el borde).

Solo se aplica cuando Método = Ajuste de plano.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>Recortar bordes de baja confianza</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Desactivado</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>Ligero (σ = 0,5 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>Medio (σ = 1 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>Fuerte (σ = 2 × step) ⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>Suavizado gaussiano del campo de deformación tras el cálculo.
σ es el ancho del núcleo gaussiano; «step» = espaciado de nodos DIC.
  Ligero  (0,5 × step): sutil, conserva detalles finos.
  Medio   (1 × step):   equilibrado, recomendado para datos ruidosos.
  Fuerte  (2 × step) ⚠: agresivo, puede difuminar gradientes reales.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>Suavizado del campo de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>Infinitesimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>Euleriana</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>Tipo de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>Recortados: %1 nodos (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>Ventana de deformación ≈ %1×%2 nodos</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ Radio VSG (%1 px) &lt; espaciado de nodos DIC (%2 px); el ajuste de plano fallará. Use VSG ≥ %3 px o cambie Método a FEM nodal.</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>Fotograma deformado</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>Fotograma de referencia</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Dibujar el campo en las posiciones deformadas de los nodos, o en sus posiciones en el fotograma de referencia.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>Mostrar en</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>Mostrar imagen de fondo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Desmarque para mostrar solo el campo, sin imagen de moteado detrás.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="82"/>
        <source>White</source>
        <translation>Blanco</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="83"/>
        <source>Black</source>
        <translation>Negro</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>Transparent</source>
        <translation>Transparente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="87"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Lo que reemplaza a la imagen cuando está oculta. Al exportar, la transparencia se conserva para PNG y TIFF; los demás formatos reciben blanco.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>Fondo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>Fondo oculto</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>Mapa de colores</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>Rango</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="122"/>
        <source>Min</source>
        <translation>Mín</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Max</source>
        <translation>Máx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="134"/>
        <source>Opacity</source>
        <translation>Opacidad</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="143"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>Rellenar bordes recortados (solo visualización)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>Reinterpola la banda de deformación recortada en los bordes a partir de nodos interiores fiables. Afecta a la vista en pantalla y a las imágenes/animaciones exportadas; los archivos de datos exportados siempre mantienen el borde recortado como NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>Bordes</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="155"/>
        <source>Strain Post-Processing</source>
        <translation>Post-procesado de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit</source>
        <translation>Ajustar</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="204"/>
        <source>Fit image to viewport</source>
        <translation>Ajustar la imagen a la vista</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoom al 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="214"/>
        <source>Zoom in</source>
        <translation>Acercar</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="272"/>
        <source>STRAIN PARAMETERS</source>
        <translation>PARÁMETROS DE DEFORMACIÓN</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="291"/>
        <source>Cancel</source>
        <translation>Cancelar</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="295"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Cancelar el cálculo de deformación en curso. Se conserva el resultado de deformación anterior.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Exportar resultados de desplazamiento y deformación a NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="340"/>
        <source>FIELD</source>
        <translation>CAMPO</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>VISUALIZATION</source>
        <translation>VISUALIZACIÓN</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="362"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNIDADES FÍSICAS</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="371"/>
        <source>LOG</source>
        <translation>REGISTRO</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="400"/>
        <source>Strain Field</source>
        <translation>Campo de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="402"/>
        <source>Analysis</source>
        <translation>Análisis</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="516"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Fallo en el cálculo de deformación: %1: %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="525"/>
        <location filename="../../gui/strain_window.py" line="587"/>
        <source>Strain computation complete.</source>
        <translation>Cálculo de deformación completado.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="536"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Ventana de deformación: no hay resultados de desplazamiento para posprocesar.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="567"/>
        <source>Cancelling…</source>
        <translation>Cancelando…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="604"/>
        <source>Strain computation cancelled.</source>
        <translation>Cálculo de deformación cancelado.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="613"/>
        <source>Strain compute failed: %1</source>
        <translation>Fallo en el cálculo de deformación: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain Computation Failed</source>
        <translation>Fallo en el cálculo de deformación</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="659"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Parámetros modificados — haga clic en «Calcular deformación»</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="220"/>
        <source>Zoom out</source>
        <translation>Alejar</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="282"/>
        <source>Compute Strain</source>
        <translation>Calcular deformación</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="304"/>
        <source>Export Results</source>
        <translation>Exportar resultados</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="547"/>
        <source>Starting…</source>
        <translation>Iniciando…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="583"/>
        <source>Complete</source>
        <translation>Completado</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>Usar unidades físicas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>Unidad: px/fotograma</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="51"/>
        <source>Incremental</source>
        <translation>Incremental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="52"/>
        <source>Accumulative</source>
        <translation>Acumulativo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>Incremental: cada fotograma se compara con el fotograma de referencia anterior.
Adecuado para grandes deformaciones acumuladas; obligatorio en grandes rotaciones.

Acumulativo: cada fotograma se compara con el fotograma 1.
Preciso solo para deformaciones pequeñas y monótonas.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>Modo de seguimiento</translation>
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
        <translation>Local DIC: Coincidencia de subconjuntos independiente (IC-GN). Rápido,
conserva detalles locales nítidos. Ideal para pequeñas
deformaciones o imágenes de alta calidad.

AL-DIC: Lagrangiano aumentado con regularización
FEM global. Impone compatibilidad de desplazamientos
entre subconjuntos. Ideal para grandes deformaciones, imágenes
con ruido o cuando la precisión de la deformación es importante.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>Solucionador</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>Cada fotograma</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>Cada N fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>Fotogramas personalizados</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>Cuándo se actualiza el fotograma de referencia durante el seguimiento incremental.
Cada fotograma: reiniciar la referencia en cada fotograma (menor desplazamiento por paso,
más robusto para grandes deformaciones).
Cada N fotogramas: reiniciar cada N fotogramas (equilibrio entre velocidad y robustez).
Fotogramas personalizados: lista de índices definida por el usuario.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>Actualización de la referencia</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>Actualizar la referencia cada N fotogramas</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>Intervalo</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>Índices de fotograma separados por comas para usar como fotogramas de referencia (base 0)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>Fotogramas de referencia</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>Arrastre la carpeta de imágenes
o Examinar</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>Seleccionar carpeta de imágenes</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>Vista previa</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>(sin imagen)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>Solo imagen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>Imagen + máscara</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>Solo máscara</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>Vista:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>Alfa:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Azul</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rojo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Verde</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Amarillo</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>Color de máscara:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>Sin máscara asignada</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>Fotograma %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>No se pudo cargar la imagen</translation>
    </message>
</context>
</TS>
