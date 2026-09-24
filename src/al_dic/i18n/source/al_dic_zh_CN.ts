<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>AL-DIC 迭代次数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>AL-DIC 求解器的全局精修迭代次数。
1 = 单次全局求解（最快），3 = 默认值，
5 次以上大多数情况下收益递减。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>仅对 AL-DIC 求解器生效，Local DIC 会忽略。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>峰值被截断时自动扩大 FFT 搜索范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>当 NCC 峰值触及搜索区域边缘时，自动以更大的搜索范围重试（最大到图像一半尺寸，每次放大 2 倍，共 6 次重试）。

仅对 FFT 初始猜测模式有效。</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="70"/>
        <source>crack</source>
        <translation>裂纹</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="72"/>
        <source>too few valid points</source>
        <translation>有效点过少</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="74"/>
        <source>unreliable (strain edge trim)</source>
        <translation>不可靠（应变边缘裁剪）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>gauge endpoint lost</source>
        <translation>量规端点失效</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="79"/>
        <source>not computed</source>
        <translation>未计算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>no data</source>
        <translation>无数据</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>线段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>圆形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>多边形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="106"/>
        <source>Click once to place a point probe.</source>
        <translation>点击一次放置点探针。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click twice: opposite corners.</source>
        <translation>点击两次：对角两点。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>点击两次：先圆心，后边缘。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="116"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>逐个点击顶点，双击闭合。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="87"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>显示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>名称</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>说明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="98"/>
        <source>Colour…</source>
        <translation>颜色…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="99"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Statistic:</source>
        <translation>统计量：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>平均值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>中位数</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="63"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>最大值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="64"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>最小值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="65"/>
        <source>Standard deviation</source>
        <translation>标准差</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="66"/>
        <source>Valid fraction</source>
        <translation>有效比例</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="50"/>
        <source>Crack opening</source>
        <translation>裂纹张开位移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="487"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>尚未计算应变。请在“应变场”页中计算，或改为绘制位移。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="531"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>量规类物理量需要线探针。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="528"/>
        <source>No visible probe can show this quantity.</source>
        <translation>没有可见的探针能显示此物理量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>no valid data: %1</source>
        <translation>无有效数据：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>crack from frame %1</source>
        <translation>第 %1 帧起出现裂纹</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="178"/>
        <source>endpoint lost from frame %1</source>
        <translation>第 %1 帧起端点失效</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="181"/>
        <source>gaps: too few valid points</source>
        <translation>有缺口：有效点过少</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="183"/>
        <source>gaps: unreliable strain</source>
        <translation>有缺口：应变不可靠</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="189"/>
        <source>not plotted: gauges need a line</source>
        <translation>未绘制：量规需要线探针</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="192"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>未绘制：单个点没有离散度或覆盖率</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="194"/>
        <source>not plotted</source>
        <translation>未绘制</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="508"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>已添加探针「%1」。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="594"/>
        <source>Clear All Probes</source>
        <translation>清除所有探针</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="595"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>删除所有探针？此操作无法撤销。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="108"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="109"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>线段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>区域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="478"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>请先运行 DIC 分析，然后才能绘制探针曲线。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="96"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>虚拟引伸计</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="98"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>裂纹量规</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>点击两次：起点和终点。线探针同时也是虚拟引伸计和裂纹张开量规。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>点击两个标距点。图表随后显示两点之间的应变。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="122"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>在裂纹两侧各点击一个点。图表随后显示裂纹的张开量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="122"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="129"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
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
        <translation>缩放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="134"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Show field</source>
        <comment>Analysis canvas: colour the image by the field</comment>
        <translation>显示场</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Colour the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>用当前帧所绘物理量的场为参考图像着色。对于量规读数，显示“应变场”页中的场。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="179"/>
        <source>Plot:</source>
        <translation>绘制：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="182"/>
        <source>X axis:</source>
        <translation>X 轴：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Strain as:</source>
        <translation>应变显示为：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Min. valid fraction:</source>
        <translation>最小有效比例：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="186"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>当线或区域中可靠点的比例低于此值时，该帧留空。避免样本逐渐缩小而曲线依然平滑的假象。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="193"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>随时间变化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>每个探针在各帧的读数。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="198"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>沿线分布</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>The field along the selected line at the current frame, over the other frames in grey.</source>
        <translation>当前帧沿所选线的场分布，其它帧以灰色显示在下方。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="205"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>时空图</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>The field along the selected line at every frame: distance against frame, value as colour.</source>
        <translation>所选线在每一帧的场分布：距离对帧，数值以颜色表示。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="212"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>应力–应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>以所绘物理量为横轴的应力（无 A0 时为载荷）曲线，每个探针一条：配合引伸计为应力–应变曲线，配合裂纹量规为载荷–张开量曲线。需要载荷数据。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="225"/>
        <source>Other frames</source>
        <translation>其它帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="226"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>在当前帧后方淡色绘制其它帧的分布（最多十二帧，均匀间隔）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="231"/>
        <source>Line data (CSV)…</source>
        <translation>线数据（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="232"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Load data…</source>
        <translation>载荷数据…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>导入试验机的载荷记录（CSV），以载荷或应力为横轴绘图，并绘制应力–应变曲线。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="239"/>
        <source>Probe data (CSV)…</source>
        <translation>探针数据（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Chart image…</source>
        <translation>图表图像…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Copy chart</source>
        <translation>复制图表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy plotted data</source>
        <translation>复制所绘数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="557"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>请通过“载荷数据…”导入试验机的载荷记录，以绘制应力–应变曲线。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="28"/>
        <source>Displacement U</source>
        <translation>位移 U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="29"/>
        <source>Displacement V</source>
        <translation>位移 V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="30"/>
        <source>Displacement magnitude</source>
        <translation>位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Extensometer strain</source>
        <translation>引伸计应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Extensometer true strain</source>
        <translation>引伸计真应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="49"/>
        <source>Elongation ΔL</source>
        <translation>伸长量 ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="51"/>
        <source>Crack sliding</source>
        <translation>裂纹滑移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="52"/>
        <source>Crack opening magnitude</source>
        <translation>裂纹张开位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="277"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="432"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="427"/>
        <source>Time (s)</source>
        <translation>时间 (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="572"/>
        <source>Load (N)</source>
        <translation>载荷 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="284"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="569"/>
        <source>Stress (MPa)</source>
        <translation>应力 (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="293"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>比值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="417"/>
        <source>Could not draw the field: %1</source>
        <translation>无法绘制场：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="134"/>
        <source>Esc cancels placement</source>
        <translation>Esc 取消放置</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="142"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>拖动可移动探针，拖动控制点可改变其形状。按 Delete 删除，按 F2 重命名。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="483"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>在参考图像上放置一个探针即可开始。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="205"/>
        <source>Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.</source>
        <translation>%1 沿线没有有效数据：其应变在边缘或孔附近被判为低置信度而裁剪。请改为绘制位移，或在“应变场”页减少裁剪。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="211"/>
        <source>Nothing valid along %1: a crack has consumed the material under it.</source>
        <translation>%1 沿线没有有效数据：其下方的材料已被裂纹消耗。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="216"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1 沿线没有有效数据：它不在测量区域内。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="608"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>沿线视图显示的是场。请选择一个场来绘制。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="612"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>放置或选择一个线探针，即可查看沿线的场。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="630"/>
        <source>Distance along %1 (%2)</source>
        <translation>沿 %1 的距离（%2）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="639"/>
        <source>%1, frame %2</source>
        <translation>%1，第 %2 帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="157"/>
        <source>not plotted: off the measured area</source>
        <translation>未绘制：不在测量区域内</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="160"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>未绘制：量规端点不在测量区域内</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="662"/>
        <source>Export Probe Data</source>
        <translation>导出探针数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <source>CSV Files</source>
        <translation>CSV 文件</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="439"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>载荷数据无法与帧对应：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <location filename="../../gui/panels/analysis/tab.py" line="722"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="691"/>
        <source>Probe export failed: %1</source>
        <translation>探针导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="693"/>
        <source>Probe data written to %1</source>
        <translation>探针数据已写入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Export Line Data</source>
        <translation>导出线数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="712"/>
        <source>Line export failed: %1</source>
        <translation>线数据导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="714"/>
        <source>Line data written to %1</source>
        <translation>线数据已写入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>SVG Images</source>
        <translation>SVG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="731"/>
        <source>Export Chart</source>
        <translation>导出图表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart copied to the clipboard.</source>
        <translation>图表已复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="758"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>所绘数据已复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="719"/>
        <source>PNG Images</source>
        <translation>PNG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="721"/>
        <source>PDF Documents</source>
        <translation>PDF 文档</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="742"/>
        <source>Chart export failed: %1</source>
        <translation>图表导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="744"/>
        <source>Chart written to %1</source>
        <translation>图表已写入 %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="983"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>为 %n 帧导入了感兴趣区域</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>请先运行 DIC —— 当前没有可后处理的位移结果。</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1103"/>
        <source>pyALDIC has hit an error</source>
        <translation>pyALDIC 发生错误</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1104"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>发生了意外错误。应用程序之后的行为可能不正常，建议保存会话并重新启动。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1114"/>
        <source>Details were written to %1</source>
        <translation>详细信息已写入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1218"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>正在后台准备计算内核。新安装后的首次分析会比之后的耗时更长。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1232"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>计算内核已就绪（%1 秒）。</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>根据每帧的数据范围自动缩放颜色范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>固定</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>所有帧都使用手动设置的最小/最大值</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批量导入感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>掩模文件夹：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>（无）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>可用掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>按文件名自动匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根据文件名中的数字把掩模文件匹配到对应帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>顺序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>从第 0 帧起按顺序把掩模分配给各帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>帧分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所选 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>将所选掩模与所选帧配对</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>选择掩模文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>无法读取掩模文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>尺寸不匹配：%1×%2（期望 %3×%4）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>%n 个掩模尺寸不匹配，已禁用。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>无效的分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>一个帧只能对应一个掩模。请选择恰好一个掩模，或者选择多个帧以把同一个掩模应用到多个帧。</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1226"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1233"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>显示网格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>显示/隐藏计算网格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>显示子集</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>悬停时显示子集窗口（需要先开启网格）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>正在放置种子点</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Mode</source>
        <translation>模式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Solver</source>
        <translation>求解器</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="43"/>
        <source>Init</source>
        <translation>初始猜测</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>累积式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="94"/>
        <source>Incremental</source>
        <translation>增量式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="101"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM（%1 次迭代）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="110"/>
        <source>Starting Points</source>
        <translation>种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>每帧 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>每 %1 帧 FFT</translation>
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
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="859"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1008"/>
        <source>Auto</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="481"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1382"/>
        <source>Opacity</source>
        <extracomment>Whether an image sits behind the field. Independent of show_deformed: which frame to use is only a question once you show one at all. Fill when the background is hidden: &quot;white&quot;, &quot;black&quot; or &quot;transparent&quot;. Fill used by images and animation alike when the background is hidden.</extracomment>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>场不透明度（0 = 透明，1 = 完全不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>全部</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>输出文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>选择输出文件夹…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>打开文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>启用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸缩放位移值，并在色条标签显示物理单位。应变为无量纲，不受影响。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="695"/>
        <source>/ pixel</source>
        <translation>/ 像素</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="697"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="712"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="714"/>
        <source>Frame rate</source>
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>动画</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>报告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>预览与色条</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 归档 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐帧）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ：逐帧一个文件（默认：合并为单个文件）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="775"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Select:</source>
        <translation>选择：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>应变</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>请先运行“计算应变”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 参数文件（JSON）始终导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>导出数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="860"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1009"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1367"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="861"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1010"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1374"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="880"/>
        <source>IMAGE SETTINGS</source>
        <translation>图像设置</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="890"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1039"/>
        <source>Format</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="898"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1047"/>
        <source>Full resolution</source>
        <translation>原始分辨率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>限制导出图像的长边（宽和高中较大的一个；保持宽高比）。
场的细节由网格密度决定，因此较小的上限几乎无损，
但文件更小、编码更快。越低越快。「原始分辨率」保持原生尺寸。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>分辨率（长边）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>变形帧：场绘制在位移后的节点位置（参考位置 + 位移），叠加在每一帧自己的照片上。
参考帧：绘制在原始节点位置，叠加在第一帧上。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>取消勾选可仅导出场，其后不含散斑图像。填充色在“预览与色条”页选择。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>限制动画的长边（宽和高中较大的一个）。
越低越快、越小。强烈建议用于 GIF：其体积在原生分辨率下会急剧膨胀。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>JPEG 质量（越高文件越大）。对 PNG/TIFF 无效。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>JPEG 质量</translation>
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
        <translation>包含色条</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>在每张图像右侧添加一条垂直色条。
启用自动范围时，刻度标签会按帧更新。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>绘制为</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>取消导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>导出图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>动画设置</translation>
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
        <translation>每 N 帧导出一帧（1 = 每帧都导出）。越大越快、越小，
但看起来更卡顿。播放时长保持不变（上方 FPS 为抽帧前的帧率）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>抽帧间隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>在每一帧右侧添加一条垂直色条。
启用自动范围时，刻度标签会按帧更新。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>取消勾选可仅导出场，其后不含散斑图像。GIF 和 MP4 无法保存透明度，透明填充将写为白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>导出动画</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>内容</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>参数摘要表</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>场统计（每帧 最小/最大/平均/标准差）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>场图像示例</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1164"/>
        <source>Sample every</source>
        <translation>每隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1170"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>位移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>应变：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>格式：HTML（自包含，可在任意浏览器中查看）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>生成报告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>打开此页以渲染预览。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>色条样式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Right</source>
        <translation>右</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Left</source>
        <translation>左</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Top</source>
        <translation>上</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Bottom</source>
        <translation>下</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1280"/>
        <source>Position</source>
        <translation>位置</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1286"/>
        <source>Font size</source>
        <translation>字号</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>字体</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>色条粗细</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1326"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1325"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1306"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1313"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>在导出内容外围加一圈空白边框，宽度为长边的比例（0 = 无）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>边距</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>边距颜色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1333"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.</source>
        <translation>关闭“显示背景图像”时，用于填充原本显示背景图像的区域。
PNG 和 TIFF 会保留透明度；JPEG、GIF 和 MP4 没有 alpha 通道，将改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>刷新预览</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>场的外观</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>应用到所有场</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>将该场的颜色映射、不透明度和自动范围应用到所有已启用的场（每个场保留各自的最小/最大值）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>预览失败：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>在“图像”页启用一个场以进行预览。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>该场/帧没有数据。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>帧范围</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>所有帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>从</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1619"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>到</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1650"/>
        <source>Select Output Folder</source>
        <translation>选择输出文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>已导出 %1 个文件 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>错误：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>开始中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>正在渲染 %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>帧 %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>已导出 %1 张图像 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>未写入任何动画。详情请查看日志。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>报告已保存 → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="18"/>
        <source>Disp U</source>
        <translation>位移 U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Disp V</source>
        <translation>位移 V</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>播放动画</translation>
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
        <translation>下一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>播放速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>帧 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>暂停动画</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>帧 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>请先加载图像，再绘制感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>三个点几乎共线 — 请在圆周上分散地选取三个点。</translation>
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
        <translation>文件名</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>区域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="190"/>
        <location filename="../../gui/widgets/image_list.py" line="248"/>
        <source>Add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="242"/>
        <source>Edit</source>
        <translation>编辑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>待绘</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>为 %n 帧导入感兴趣区域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>清除感兴趣区域（%1 帧已有区域）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>清除感兴趣区域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>删除 %n 张图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>已选择 %1 个文件用于 %2 帧 — 数量必须匹配</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>放置若干种子点；pyALDIC 在每个点上运行单点 NCC 引导，然后沿网格邻居传播位移场。

最适合：
• 大帧间位移（&gt; 50 px）
• 不连续场（裂纹、剪切带）
• FFT 容易选错峰的场景

绘制或编辑 ROI 时会为每个区域自动放置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>放置种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>在画布上进入放置模式。左键添加、右键删除，按 Esc 或再次点击退出。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>自动放置</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>在每个空区域填入 NCC 最高的节点。已有种子点会保留。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>移除所有种子点。比逐个右键删除快。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 区域就绪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT（互相关）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>全网格归一化互相关。在搜索半径内稳健；峰值被截断时搜索自动扩展。

最适合：
• 小到中等的平滑运动
• 纹理良好的散斑
• 不需要用户额外设置

计算成本随搜索半径增长，极大位移会变慢。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>每</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>每 N 帧运行一次 FFT。N = 1 表示每帧都做 FFT（最安全但最慢）。N &gt; 1 在两次重置之间使用热启动，将误差传播限制在 N 帧内。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>（N=1 即每帧）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>仅在参考帧更新时（只对增量模式）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>参考帧变化时运行 FFT；每段内使用热启动。是增量模式的典型默认值。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>使用前一帧收敛的位移作为初始猜测。不运行任何互相关。

最适合：
• 非常小的帧间运动（几像素）
• 运动平滑时速度最快

长序列中误差会累积。数据有噪声或运动较大时请选 FFT 或种子点。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>放置中…（再次点击退出）</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="185"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>自然排序（1, 2, …, 10）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="188"/>
        <source>Sort by embedded numbers: image1, image2, …, image10
Default (unchecked): lexicographic — best for zero-padded names</source>
        <translation>按文件名中的数字排序：image1, image2, …, image10
默认（不勾选）：字典序 — 适合已补零的文件名</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>工作流类型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>初始猜测</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>感兴趣区域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>高级</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>未选择文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>选择文件…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>载荷列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>按时间</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>按帧号</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>行与帧的对应方式：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>时间列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>参考图像时刻对应的试验机时间。若相机比试验机晚 2 s 启动，则输入 2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>偏移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>帧号列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>第一张图像的编号：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>初始横截面积，用于计算工程应力 F / A0（MPa）。仅需载荷时保持为 0。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>横截面积 A0：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>移除载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>相机帧率：%1 fps，取自“物理单位”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>请在“物理单位”中设置相机帧率，才能按时间对应。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>（未命名）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>没有帧落在记录范围内：请检查所选列和偏移。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>有载荷值的帧：%1 / %2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>打开载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV 文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>无法读取 %1：%2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>打开会话…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>保存会话…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>将 .aldic 文件关联到 pyALDIC…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>注册 .aldic，让双击会话文件即可打开 pyALDIC（仅当前用户，无需管理员权限）。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>语言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>语言已切换</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>语言已切换至 %1。请重启 pyALDIC 以让所有界面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>保存会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC 会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>较大</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>包含结果？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>在此会话中包含已计算的结果吗？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>包含结果（未压缩约 %1）可让你下次直接打开会话而无需重新计算。选择“否”则只保存一个小的仅配置文件，便于分享。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>正在保存会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>保存会话失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>打开会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>正在加载会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>打开会话失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>定位会话图像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>未找到此会话保存的图像文件夹:
%1

结果已恢复。要显示背景图像,请选择现在包含这些图像的文件夹。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>选择图像文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>文件关联失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>无法注册 .aldic 文件：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>文件关联</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>完成。现在双击 .aldic 文件即可打开 pyALDIC 并恢复该会话。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>帧 %1 没有自己的感兴趣区域 — 计算时使用帧 1 的感兴趣区域。请切换到帧 1 编辑，或导入掩模为此帧单独指定。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>没有可保存的感兴趣区域 — 请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>感兴趣区域掩模为空。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>保存感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>PNG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>掩模已保存至 %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>没有可反选的感兴趣区域 — 请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>请先在帧 1 上定义感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  已导入帧 %1 的掩模</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>批量导入：已加载 %n 个掩模</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>网格颜色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>点击选择网格线颜色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>线宽</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="48"/>
        <source>Subset Size</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past a cap of about 28 characters of the label font, wrapped rather than squeezing the inputs.</extracomment>
        <translation>子集尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="54"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>IC-GN 子集窗口尺寸（像素，奇数）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <source>Subset Step</source>
        <translation>子集步长</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="64"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>节点间距（像素，必须是 2 的幂）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="72"/>
        <location filename="../../gui/widgets/param_panel.py" line="197"/>
        <location filename="../../gui/widgets/param_panel.py" line="209"/>
        <source>Search Range</source>
        <translation>搜索范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="87"/>
        <source>Refine Inner Boundary</source>
        <translation>加密内部边界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>沿内部掩模边界局部加密网格
（感兴趣区域内部的孔洞）。适合气泡 / 空洞边缘。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="94"/>
        <source>Refine Outer Boundary</source>
        <translation>加密外部边界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="97"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>沿感兴趣区域的外部边界局部加密网格。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="112"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>加密强度。最小单元尺寸 = max(2, subset_step / 2^level)。对内部边界、外部边界和画笔加密区域统一生效。可用级别取决于子集尺寸和步长。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="118"/>
        <source>Refinement Level</source>
        <translation>加密级别</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>FFT 搜索可检测到的每帧最大位移（像素）。
设置值应略大于预期的帧间运动。
对于增量模式下的大旋转，该值必须覆盖
  半径 × sin(单步角度)。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="185"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>每个种子点处单点 NCC 搜索的初始半宽（像素）。
若峰值被截断，每次重试自动放大 2 倍，最大到图像一半尺寸。
仅影响种子点引导；其他节点使用 F-aware 传播（无需逐节点搜索）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="193"/>
        <location filename="../../gui/widgets/param_panel.py" line="209"/>
        <source>Starting Point Search</source>
        <translation>种子点搜索</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="249"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>轻度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>中等</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="251"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>强</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="252"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>超强</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="253"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>极限</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="259"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="281"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>最小单元尺寸 = %1 px  (subset_step=%2, level=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>使用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>单个图像像素对应的物理尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="80"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>采集帧率（用于速度场）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>位移：%1  速度：%2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/帧</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>正在构建流水线配置…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>正在加载图像…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  已加载 %1 张图像，shape=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  ROI 掩模：%1，%2 像素（%3%）</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>已取消运行：请为缺失的参考帧定义逐帧感兴趣区域，或在下次运行时接受继承自第 1 帧的掩模。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n 帧使用自定义 ROI 掩模</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>已收到结果：%n 帧</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>开始 DIC 分析…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>分析完成，用时 %1 秒</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>用户已停止分析。</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>请先加载图像，再在第 1 帧上绘制感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累积模式&lt;/b&gt; — 只有第 1 帧需要感兴趣区域。后续帧都直接与其比较。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;增量模式，每帧&lt;/b&gt; — 第 1 帧需要感兴趣区域。系统会自动将其扭曲到每个后续帧（无需逐帧绘制）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，每 %1 帧&lt;/b&gt; — 请在以下帧绘制感兴趣区域：&lt;b&gt;%2&lt;/b&gt;（共 %3 个参考帧）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;增量模式，自定义&lt;/b&gt; — 未设置自定义参考帧。仅第 1 帧为参考；请在参考帧列表中添加更多索引。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，自定义&lt;/b&gt; — 请在以下帧绘制感兴趣区域：&lt;b&gt;%1&lt;/b&gt;（共 %2 个参考帧）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>请在第 1 帧绘制感兴趣区域。</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 添加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>向感兴趣区域添加形状（多边形 / 矩形 / 圆形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>从感兴趣区域裁剪形状（多边形 / 矩形 / 圆形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 加密</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>用画笔绘制额外的网格加密区域
（仅在第 1 帧可用 — 网格点会自动扭曲到后续帧）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>加密画笔仅在第 1 帧可用。切换到第 1 帧后可绘制加密区域；系统会自动将其扭曲到后续帧。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>导入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>从图像文件导入掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>批量导入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>批量导入多帧的掩模文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>将当前掩模保存为 PNG 文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>反选</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>反选感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>清除所有感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>绘制</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>擦除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>清除画笔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>多边形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>圆形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>圆形（三点）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>Import Mask Image</source>
        <translation>导入掩模图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="356"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>运行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>取消当前分析。已计算的帧会被保留，你可以查看或导出这部分结果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>打开应变窗口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在独立的后处理窗口中计算并可视化应变。需先完成一次运行以获得位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>进度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>就绪</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>已用时间  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>剩余时间  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>显示于</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>将场绘制在变形后的节点位置，或绘制在其在参考帧中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾选可仅显示场，其后不显示散斑图像。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="178"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="183"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景隐藏时用什么替代图像。导出时 PNG 和 TIFF 会保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>叠加图不透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>运行前请在每个红色区域放置至少一个种子点（红色 = 需要种子点）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  第 %2 帧</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="150"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="161"/>
        <source>STRAIN</source>
        <translation>应变</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>播放动画</translation>
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
        <translation>下一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>播放速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>帧 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>暂停动画</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>帧 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>平面拟合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>FEM 节点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Method</source>
        <translation>方法</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="102"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>VSG（虚拟应变计，Virtual Strain Gauge）尺寸指围绕每个网格节点、用于拟合局部位移平面的圆形区域的直径（像素）。应变由该平面的斜率给出。

• VSG 越大 → 应变越平滑，空间分辨率越低。
• VSG 越小 → 应变越锐利，但噪声越大。
• 经验法则：VSG ≥ 2 × 子集步长 + 1（默认：41 px）。

方法选择 FEM 节点时不使用此参数（此时由网格间距决定虚拟应变计尺寸）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>VSG 尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>均匀网格下圆形 VSG 窗口内每个轴向的网格节点数：2 × floor(VSG 半径 / 节点间距) + 1。平面拟合使用半径内的所有节点；在加密网格上该数量会局部变化。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>在 ROI / 孔洞边缘隐藏低置信度的应变：那里 VSG 窗口跨越边界，局部平面拟合变成单边、不可靠。

• 系数 × VSG 半径 = 裁剪边界带的宽度。
• 0.00 = 保留所有节点（不裁剪）。
• 0.70 = 推荐（裁掉误差明显上升的区域）。
• 1.00 = 最严格（窗口一旦触及边界即裁剪）。

仅在 方法 = 平面拟合 时生效。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>裁剪低置信度边缘</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>轻度（σ = 0.5 × step）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>中等（σ = 1 × step）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>强（σ = 2 × step）⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>计算后对应变场做高斯平滑。
σ 为高斯核宽度；“step” 为 DIC 节点间距。
  轻度（0.5 × step）：轻度平滑，保留细节。
  中等（1 × step）：平衡选择，推荐用于噪声数据。
  强（2 × step）⚠：强平滑，可能模糊真实梯度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>应变场平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>无穷小应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>欧拉应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>格林-拉格朗日应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>应变类型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>已裁剪：%1 个节点 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>应变窗口 ≈ %1×%2 节点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ VSG 半径（%1 px）&lt; DIC 节点间距（%2 px）；平面拟合将失败。请将 VSG ≥ %3 px 或将方法切换为 FEM 节点。</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>将场绘制在变形后的节点位置，或绘制在其在参考帧中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>显示于</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾选可仅显示场，其后不显示散斑图像。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="82"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="83"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="87"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景隐藏时用什么替代图像。导出时 PNG 和 TIFF 会保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="122"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="134"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="143"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>填充裁剪的边缘（仅显示）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>从可靠的内部节点重新插值被边缘裁剪的应变带。影响屏幕显示和导出的图像/动画；导出的数据文件始终将裁剪的边缘保留为 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>边缘</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="155"/>
        <source>Strain Post-Processing</source>
        <translation>应变后处理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit</source>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="204"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="214"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="272"/>
        <source>STRAIN PARAMETERS</source>
        <translation>应变参数</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="291"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="295"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>取消正在进行的应变计算。保留之前的应变结果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>将位移和应变结果导出为 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="340"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="362"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="371"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="400"/>
        <source>Strain Field</source>
        <translation>应变场</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="402"/>
        <source>Analysis</source>
        <translation>分析</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="516"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>应变计算失败：%1：%2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="525"/>
        <location filename="../../gui/strain_window.py" line="587"/>
        <source>Strain computation complete.</source>
        <translation>应变计算完成。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="536"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>应变窗口：没有可后处理的位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="567"/>
        <source>Cancelling…</source>
        <translation>正在取消…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="604"/>
        <source>Strain computation cancelled.</source>
        <translation>应变计算已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="613"/>
        <source>Strain compute failed: %1</source>
        <translation>应变计算失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain Computation Failed</source>
        <translation>应变计算失败</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="659"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 参数已变更 — 请点击“计算应变”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="220"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="282"/>
        <source>Compute Strain</source>
        <translation>计算应变</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="304"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="547"/>
        <source>Starting…</source>
        <translation>开始中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="583"/>
        <source>Complete</source>
        <translation>已完成</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>使用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>单位：px/帧</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="51"/>
        <source>Incremental</source>
        <translation>增量式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="52"/>
        <source>Accumulative</source>
        <translation>累积式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>增量式：每帧与前一个参考帧比较。
适用于大量累积变形，大旋转场景必须使用。

累积式：每帧都与第 1 帧比较。
仅适用于小的、单调的变形。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>追踪模式</translation>
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
        <translation>Local DIC：独立子集匹配（IC-GN）。速度快，
保留局部锐利特征。适合小变形
或高质量图像。

AL-DIC：全局 FEM 正则化的增广拉格朗日方法。
强制子集间的位移相容性。适合大变形、
噪声图像，或对应变精度要求高的场景。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>求解器</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>每帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>每 N 帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>自定义帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>增量追踪中参考帧的刷新策略。
每帧：每帧都更新参考（单步位移最小，
对大变形最稳健）。
每 N 帧：每 N 帧更新一次（速度与稳健性的折中）。
自定义帧：由用户指定参考帧索引列表。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>参考帧更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>每 N 帧更新一次参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>间隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>用作参考帧的帧索引列表（0 为起始），用逗号分隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>参考帧列表</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>拖入图像文件夹
或点击浏览</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>选择图像文件夹</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>（无图像）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>仅图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>图像 + 掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>仅掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>视图：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>不透明度：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>红色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>黄色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>掩模颜色：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>未分配掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>帧 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>无法加载图像</translation>
    </message>
</context>
</TS>
