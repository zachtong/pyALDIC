<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ko" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>AL-DIC 반복 횟수</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>AL-DIC 솔버의 전역 세분화 반복 횟수.
1 = 단일 패스(가장 빠름), 3 = 기본값,
5 이상은 대부분의 경우 수익이 감소합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>AL-DIC 솔버에만 적용됩니다. Local DIC에서는 무시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>피크가 잘리면 FFT 탐색 자동 확장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>NCC 피크가 탐색 영역 경계에 도달하면 더 넓은 영역으로 자동 재시도합니다(이미지 절반 크기까지, 2배씩 6회).

FFT 초기 추정 모드에만 관련됩니다.</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="70"/>
        <source>crack</source>
        <translation>균열</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="72"/>
        <source>too few valid points</source>
        <translation>유효한 점이 너무 적음</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="74"/>
        <source>unreliable (strain edge trim)</source>
        <translation>신뢰할 수 없음(변형률 가장자리 트림)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>gauge endpoint lost</source>
        <translation>게이지 끝점 상실</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="79"/>
        <source>not computed</source>
        <translation>계산되지 않음</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>no data</source>
        <translation>데이터 없음</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>점</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>선분</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>사각형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>원</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>다각형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="106"/>
        <source>Click once to place a point probe.</source>
        <translation>한 번 클릭하여 점 프로브를 배치합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click twice: opposite corners.</source>
        <translation>두 번 클릭: 마주 보는 두 모서리.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>두 번 클릭: 중심, 그다음 가장자리.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="116"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>각 꼭짓점을 클릭한 뒤 두 번 클릭하여 닫습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="87"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>이름</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>종류</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>색상</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>비고</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="98"/>
        <source>Colour…</source>
        <translation>색상…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="99"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>삭제</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Clear All</source>
        <translation>모두 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="178"/>
        <source>Statistic:</source>
        <translation>통계량:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>평균</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>중앙값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="63"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>최댓값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="64"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>최솟값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="65"/>
        <source>Standard deviation</source>
        <translation>표준편차</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="66"/>
        <source>Valid fraction</source>
        <translation>유효 비율</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="50"/>
        <source>Crack opening</source>
        <translation>균열 개구 변위</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="484"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>변형률이 아직 계산되지 않았습니다. &apos;변형률장&apos; 탭에서 계산하거나 변위를 표시하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="528"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>게이지 양에는 선 프로브가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="525"/>
        <source>No visible probe can show this quantity.</source>
        <translation>이 양을 표시할 수 있는 보이는 프로브가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>no valid data: %1</source>
        <translation>유효한 데이터 없음: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>crack from frame %1</source>
        <translation>%1 프레임부터 균열</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="178"/>
        <source>endpoint lost from frame %1</source>
        <translation>%1 프레임부터 끝점 상실</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="181"/>
        <source>gaps: too few valid points</source>
        <translation>공백: 유효 점이 너무 적음</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="183"/>
        <source>gaps: unreliable strain</source>
        <translation>공백: 변형률 신뢰 불가</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="189"/>
        <source>not plotted: gauges need a line</source>
        <translation>표시 안 됨: 게이지에는 선이 필요</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="192"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>표시 안 됨: 한 점에는 분산이나 적용 범위가 없음</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="194"/>
        <source>not plotted</source>
        <translation>표시 안 됨</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="508"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>프로브 &apos;%1&apos;을(를) 추가했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="594"/>
        <source>Clear All Probes</source>
        <translation>모든 프로브 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="595"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>모든 프로브를 삭제하시겠습니까? 되돌릴 수 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="108"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>점</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="109"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>선분</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>영역</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="475"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>프로브를 그리려면 먼저 DIC 분석을 실행하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="96"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>가상 신율계</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="98"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>균열 게이지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>두 번 클릭: 시작점과 끝점. 선은 가상 신율계이자 균열 개구 게이지이기도 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>두 개의 표점을 클릭하십시오. 차트에 두 점 사이의 변형률이 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="122"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>균열 양쪽에 한 점씩 클릭하십시오. 차트에 균열이 벌어진 정도가 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="122"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="129"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
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
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="134"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Show field</source>
        <comment>Analysis canvas: colour the image by the field</comment>
        <translation>필드 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Colour the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>현재 프레임에서 표시 중인 양의 필드로 기준 이미지를 색칠합니다. 게이지 판독값의 경우 &apos;변형률장&apos; 탭의 필드를 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="176"/>
        <source>Plot:</source>
        <translation>표시:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="179"/>
        <source>X axis:</source>
        <translation>X 축:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="180"/>
        <source>Strain as:</source>
        <translation>변형률 표시:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="182"/>
        <source>Min. valid fraction:</source>
        <translation>최소 유효 비율:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>선이나 영역에서 신뢰할 수 있는 점의 비율이 이 값보다 낮으면 해당 프레임은 비워 둡니다. 표본이 줄어드는데도 곡선이 매끄럽게 유지되는 것을 방지합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="190"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>시간 변화</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="192"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>각 프로브의 모든 프레임에서의 판독값.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>선을 따른 분포</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>The field along the selected line at the current frame, over the other frames in grey.</source>
        <translation>현재 프레임에서 선택한 선을 따른 필드이며, 다른 프레임은 뒤에 회색으로 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>키모그래프</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="204"/>
        <source>The field along the selected line at every frame: distance against frame, value as colour.</source>
        <translation>모든 프레임에서 선택한 선을 따른 필드: 거리 대 프레임, 값은 색으로 표시.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>응력–변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="211"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>표시 중인 양에 대한 응력(A0가 없으면 하중) 곡선을 프로브마다 그립니다: 신율계면 응력–변형률, 균열 게이지면 하중–개구 곡선입니다. 하중 데이터가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="222"/>
        <source>Other frames</source>
        <translation>다른 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="223"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>현재 프레임 뒤에 다른 프레임의 분포를 흐리게 그립니다(최대 12개, 균등 간격).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Line data (CSV)…</source>
        <translation>선 데이터(CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="229"/>
        <source>Export</source>
        <translation>내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="230"/>
        <source>Load data…</source>
        <translation>하중 데이터…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="231"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>시험기의 하중 기록(CSV)을 가져와 하중 또는 응력에 대해 그래프를 그리고 응력–변형률 곡선을 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Probe data (CSV)…</source>
        <translation>프로브 데이터(CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="238"/>
        <source>Chart image…</source>
        <translation>차트 이미지…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="240"/>
        <source>Copy chart</source>
        <translation>차트 복사</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="242"/>
        <source>Copy plotted data</source>
        <translation>표시된 데이터 복사</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="554"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>응력–변형률 곡선을 그리려면 &apos;하중 데이터…&apos;로 시험기의 하중 기록을 가져오십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="28"/>
        <source>Displacement U</source>
        <translation>변위 U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="29"/>
        <source>Displacement V</source>
        <translation>변위 V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="30"/>
        <source>Displacement magnitude</source>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Extensometer strain</source>
        <translation>신율계 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Extensometer true strain</source>
        <translation>신율계 진변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="49"/>
        <source>Elongation ΔL</source>
        <translation>신장량 ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="51"/>
        <source>Crack sliding</source>
        <translation>균열 미끄럼</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="52"/>
        <source>Crack opening magnitude</source>
        <translation>균열 개구 변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="274"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="276"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="424"/>
        <source>Time (s)</source>
        <translation>시간 (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="278"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="426"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="569"/>
        <source>Load (N)</source>
        <translation>하중 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="428"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="566"/>
        <source>Stress (MPa)</source>
        <translation>응력 (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="290"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>비율</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="417"/>
        <source>Could not draw the field: %1</source>
        <translation>필드를 그릴 수 없습니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="134"/>
        <source>Esc cancels placement</source>
        <translation>Esc 키로 배치 취소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="142"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>드래그하여 프로브를 이동하거나 핸들을 드래그하여 모양을 바꿉니다. Delete로 삭제하고 F2로 이름을 바꿉니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>기준 이미지에 프로브를 배치하면 시작됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="205"/>
        <source>Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 가장자리나 구멍 근처의 변형률이 저신뢰도로 잘려 나갔습니다. 변위를 표시하거나 &apos;변형률장&apos; 탭에서 잘라내기를 줄이십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="211"/>
        <source>Nothing valid along %1: a crack has consumed the material under it.</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 아래의 재료가 균열로 소실되었습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="216"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 측정 영역 밖에 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="605"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>선 보기는 필드를 표시합니다. 표시할 필드를 선택하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="609"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>선 프로브를 배치하거나 선택하면 선을 따른 필드를 볼 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="627"/>
        <source>Distance along %1 (%2)</source>
        <translation>%1을(를) 따른 거리(%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="636"/>
        <source>%1, frame %2</source>
        <translation>%1, %2 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="157"/>
        <source>not plotted: off the measured area</source>
        <translation>표시 안 됨: 측정 영역 밖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="160"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>표시 안 됨: 게이지 끝점이 측정 영역 밖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="662"/>
        <source>Export Probe Data</source>
        <translation>프로브 데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <source>CSV Files</source>
        <translation>CSV 파일</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="439"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>하중 데이터를 프레임에 맞출 수 없습니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="655"/>
        <location filename="../../gui/panels/analysis/tab.py" line="722"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="691"/>
        <source>Probe export failed: %1</source>
        <translation>프로브 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="693"/>
        <source>Probe data written to %1</source>
        <translation>프로브 데이터를 %1에 기록했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Export Line Data</source>
        <translation>선 데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="712"/>
        <source>Line export failed: %1</source>
        <translation>선 데이터 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="714"/>
        <source>Line data written to %1</source>
        <translation>선 데이터를 %1에 기록했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>SVG Images</source>
        <translation>SVG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="731"/>
        <source>Export Chart</source>
        <translation>차트 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart copied to the clipboard.</source>
        <translation>차트를 클립보드에 복사했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="758"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>표시된 데이터를 클립보드에 복사했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="719"/>
        <source>PNG Images</source>
        <translation>PNG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="721"/>
        <source>PDF Documents</source>
        <translation>PDF 문서</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="742"/>
        <source>Chart export failed: %1</source>
        <translation>차트 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="744"/>
        <source>Chart written to %1</source>
        <translation>차트를 %1에 기록했습니다</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="983"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>%n 프레임에 관심 영역 가져옴</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>DIC를 먼저 실행하세요 —— 후처리할 변위 결과가 없습니다.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1094"/>
        <source>pyALDIC has hit an error</source>
        <translation>pyALDIC에서 오류가 발생했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1095"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>예기치 않은 오류가 발생했습니다. 이후 애플리케이션이 정상적으로 동작하지 않을 수 있으므로 세션을 저장하고 다시 시작하는 것을 권장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1105"/>
        <source>Details were written to %1</source>
        <translation>자세한 내용을 %1에 기록했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1209"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>백그라운드에서 계산 커널을 준비하고 있습니다. 새로 설치한 후 첫 번째 분석은 이후보다 오래 걸립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1223"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>계산 커널 준비 완료(%1초).</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>각 프레임의 데이터 범위에 맞춰 색상 범위를 다시 조정합니다</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>고정</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>모든 프레임에서 수동 최소/최대 값을 유지합니다</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>관심 영역 마스크 일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>마스크 폴더:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>(없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>찾아보기…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>사용 가능한 마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>이름으로 자동 매칭</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>파일명의 숫자로 마스크 파일을 프레임에 매칭합니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>순차 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>프레임 0부터 순서대로 마스크를 프레임에 할당합니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>프레임 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>선택 항목 할당 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>선택한 마스크와 선택한 프레임을 짝지웁니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>모두 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>마스크 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>마스크 파일을 읽지 못했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>형상 불일치: %1×%2 (예상 %3×%4)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>크기가 일치하지 않는 마스크 %n개가 비활성화되었습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>잘못된 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>한 프레임에는 마스크를 하나만 지정할 수 있습니다. 마스크를 정확히 하나 선택하거나, 여러 프레임을 선택해 하나의 마스크를 여러 프레임에 지정하세요.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1226"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1233"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>격자 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>계산 메시 격자 표시/숨김</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>서브셋 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>마우스 오버 시 서브셋 창 표시(격자 필요)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>시드점 배치 중</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Mode</source>
        <translation>모드</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Solver</source>
        <translation>솔버</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="43"/>
        <source>Init</source>
        <translation>초기값</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>누적형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="94"/>
        <source>Incremental</source>
        <translation>증분형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="101"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM(%1회 반복)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="110"/>
        <source>Starting Points</source>
        <translation>시드점</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>매 프레임 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>%1 프레임마다 FFT</translation>
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
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="859"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1008"/>
        <source>Auto</source>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="481"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1382"/>
        <source>Opacity</source>
        <extracomment>Whether an image sits behind the field. Independent of show_deformed: which frame to use is only a question once you show one at all. Fill when the background is hidden: &quot;white&quot;, &quot;black&quot; or &quot;transparent&quot;. Fill used by images and animation alike when the background is hidden.</extracomment>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>필드 불투명도 (0 = 투명, 1 = 완전 불투명)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>모두 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>모두 해제</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>출력 폴더</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>출력 폴더 선택…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>찾아보기…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>폴더 열기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>물리 단위 활성화</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>변위 값을 픽셀 크기로 스케일링하고 색상 막대 레이블에 물리 단위를 표시합니다. 변형률은 무차원이므로 영향받지 않습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="695"/>
        <source>/ pixel</source>
        <translation>/ 픽셀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="697"/>
        <source>Pixel size</source>
        <translation>픽셀 크기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="712"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="714"/>
        <source>Frame rate</source>
        <translation>프레임 속도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>데이터</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>애니메이션</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>보고서</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>미리보기 및 컬러바</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>형식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 아카이브 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV(프레임별)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ: 프레임별 1 파일(기본값: 통합 단일 파일)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="775"/>
        <source>DISPLACEMENT</source>
        <translation>변위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="784"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="811"/>
        <source>Select:</source>
        <translation>선택:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>변형률</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>먼저 「변형률 계산」을 실행하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 매개변수 파일(JSON)은 항상 내보내집니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>색상 맵</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="860"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1009"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1367"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="861"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1010"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1374"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="880"/>
        <source>IMAGE SETTINGS</source>
        <translation>이미지 설정</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="890"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1039"/>
        <source>Format</source>
        <translation>형식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="898"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1047"/>
        <source>Full resolution</source>
        <translation>전체 해상도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>내보내는 이미지의 긴 변(너비/높이 중 큰 값, 종횡비 유지)을 제한합니다.
필드 세부 정보는 메시로 결정되므로 상한을 낮춰도 거의 무손실이며,
파일이 작고 인코딩이 빠릅니다. 낮을수록 빠름. &apos;전체 해상도&apos;는 원본 크기를 유지합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>해상도(긴 변)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>변형된 프레임: 필드를 변위된 노드 위치(참조 위치 + 변위)에 그려 각 프레임 자체의 사진 위에 겹칩니다.
참조 프레임: 원래 노드 위치에 그려 첫 번째 프레임 위에 겹칩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. 채우기는 Preview &amp; Colorbar 탭에서 선택합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>애니메이션의 긴 변(너비/높이 중 큰 값)을 제한합니다.
낮을수록 빠르고 작습니다. GIF에 강력히 권장됩니다. 원본 해상도에서는 크기가 급격히 커집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>JPEG 품질(높을수록 파일이 커집니다). PNG/TIFF에서는 무시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>JPEG 품질</translation>
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
        <translation>컬러바 포함</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>각 이미지 오른쪽에 수직 컬러바를 추가합니다.
자동 범위가 활성화되면 눈금 레이블이 프레임별로 갱신됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>렌더링 방식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>내보내기 취소</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>이미지 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>애니메이션 설정</translation>
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
        <translation>N 프레임마다 하나씩 내보냅니다(1 = 모든 프레임). 클수록 빠르고 작지만,
더 끊겨 보입니다. 재생 시간은 유지됩니다(위의 FPS는 추출 전 프레임률).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>프레임 간격</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>각 프레임 오른쪽에 수직 컬러바를 추가합니다.
자동 범위가 활성화되면 눈금 레이블이 프레임별로 갱신됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. GIF와 MP4는 투명도를 저장할 수 없으므로 투명 채우기는 흰색으로 기록됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>애니메이션 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>내용</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>매개변수 요약 표</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>필드 통계 (프레임별 최소/최대/평균/표준편차)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>필드 이미지 샘플</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1164"/>
        <source>Sample every</source>
        <translation>샘플 간격</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1170"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>변위:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>변형률:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>형식: HTML (자체 포함, 모든 브라우저에서 볼 수 있음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>보고서 생성</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>이 탭을 열면 미리보기가 렌더링됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>컬러바 스타일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Right</source>
        <translation>오른쪽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1276"/>
        <source>Left</source>
        <translation>왼쪽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Top</source>
        <translation>위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1277"/>
        <source>Bottom</source>
        <translation>아래</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1280"/>
        <source>Position</source>
        <translation>위치</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1286"/>
        <source>Font size</source>
        <translation>글꼴 크기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>글꼴</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>막대 두께</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1326"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1303"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1319"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1325"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1306"/>
        <source>Background</source>
        <translation>배경</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1313"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>내보내는 콘텐츠 주위에 여백 테두리를 추가합니다. 너비는 긴 변에 대한 비율입니다(0 = 없음).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>여백</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>여백 색상</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1327"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1333"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.</source>
        <translation>‘배경 이미지 표시’를 끄면 배경 이미지가 있던 자리를 채우는 색입니다.
PNG와 TIFF는 투명도를 유지합니다. JPEG, GIF, MP4는 알파 채널이 없어 흰색으로 대체됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>미리보기 새로고침</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>필드 모양</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>모든 필드에 적용</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>이 필드의 colormap, 불투명도, 자동 범위를 활성화된 모든 필드에 적용합니다(각 필드의 min/max는 유지).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>미리보기 실패: </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>미리보려면 Images 탭에서 필드를 활성화하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>이 필드/프레임에 데이터가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>프레임 범위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>모든 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>시작</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1619"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>끝</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1650"/>
        <source>Select Output Folder</source>
        <translation>출력 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>%1 개 파일 내보냄 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>오류: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>%1 렌더링 중 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>프레임 %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>%1 개 이미지 내보냄 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>애니메이션이 기록되지 않았습니다. 자세한 내용은 로그를 확인하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>보고서 저장됨 → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="18"/>
        <source>Disp U</source>
        <translation>변위 U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Disp V</source>
        <translation>변위 V</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>애니메이션 재생</translation>
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
        <translation>다음 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>재생 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>프레임 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>애니메이션 일시정지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>프레임 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>관심 영역을 그리기 전에 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>세 점이 거의 일직선입니다 — 원의 가장자리에 고르게 세 점을 찍으세요.</translation>
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
        <translation>파일 이름</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>영역</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="190"/>
        <location filename="../../gui/widgets/image_list.py" line="248"/>
        <source>Add</source>
        <translation>추가</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="242"/>
        <source>Edit</source>
        <translation>편집</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>필요</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>%n 프레임에 관심 영역 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>관심 영역 지우기 (%1개 프레임에 영역 있음)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>관심 영역 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>%n 개 이미지 삭제</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%2 프레임에 대해 %1 개 파일 선택됨 — 개수가 일치해야 합니다</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>시드점</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>몇 개의 시드점을 배치합니다. pyALDIC은 단점 NCC로 각각을 초기화하고 메시 이웃을 따라 필드를 전파합니다.

적합한 경우:
• 큰 프레임 간 변위(&gt; 50 px)
• 불연속 필드(균열, 전단대)
• FFT가 잘못된 피크를 고르는 경우

ROI 생성/편집 시 영역별로 자동 배치됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>시드점 배치</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>캔버스에서 배치 모드로 들어갑니다. 좌클릭으로 추가, 우클릭으로 제거, Esc 또는 다시 클릭하여 종료합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>자동 배치</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>각 빈 영역에 최고 NCC 노드를 배치합니다. 기존 시드점은 유지됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>모든 시드점을 제거합니다. 하나씩 우클릭하는 것보다 빠릅니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 영역 준비 완료</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT(상호상관)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>전체 격자 정규화 상호상관. 탐색 반경 내에서 견고하며, 피크가 잘리면 탐색이 자동 확장됩니다.

적합한 경우:
• 작거나 중간 크기의 부드러운 움직임
• 질감이 좋은 스페클
• 특별한 설정이 불필요

비용이 탐색 반경과 함께 증가하므로 매우 큰 변위에서는 느려집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>매</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>N 프레임마다 FFT를 실행합니다. N = 1은 매 프레임 FFT(가장 안전·가장 느림). N &gt; 1은 리셋 사이에 웜스타트를 사용해 오류 전파를 N 프레임 이내로 제한합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>(N=1 은 매 프레임)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>기준 프레임 갱신 시에만(증분형만)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>기준 프레임이 바뀔 때마다 FFT를 실행하고, 각 구간 내에서는 웜스타트를 사용합니다. 증분 모드의 표준 기본값입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>이전 프레임의 수렴된 변위를 초기 추정으로 사용합니다. 상호상관을 실행하지 않습니다.

적합한 경우:
• 매우 작은 프레임 간 움직임(몇 픽셀)
• 움직임이 부드러울 때 가장 빠른 옵션

긴 시퀀스에서 오류가 누적될 수 있습니다. 노이즈 데이터나 움직임이 클 때는 FFT 또는 시드점을 권장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>배치 중…(클릭하여 종료)</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="185"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>자연 정렬 (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="188"/>
        <source>Sort by embedded numbers: image1, image2, …, image10
Default (unchecked): lexicographic — best for zero-padded names</source>
        <translation>파일명 속 숫자 순 정렬: image1, image2, …, image10
기본(체크 해제): 사전식 — 0 채움 파일명에 적합</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>워크플로 유형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>초기 추정</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>관심 영역</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>매개변수</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>고급 설정</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>하중 데이터</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>선택한 파일이 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>파일 선택…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>하중 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>시간으로</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>프레임 번호로</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>행과 프레임 연결:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>시간 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>기준 이미지 시점의 시험기 시간입니다. 카메라가 시험기보다 2 s 늦게 시작했다면 2를 입력하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>오프셋:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>프레임 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>첫 이미지의 번호:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>초기 단면적으로, 공칭 응력 F / A0(MPa) 계산에 사용합니다. 하중만 필요하면 0으로 두십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>단면적 A0:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>하중 데이터 제거</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>카메라 프레임 속도: %1 fps, &apos;물리 단위&apos;에서 설정.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>시간으로 맞추려면 &apos;물리 단위&apos;에서 카메라 프레임 속도를 설정하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(이름 없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>기록 범위에 드는 프레임이 없습니다: 열과 오프셋을 확인하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>하중이 있는 프레임: %1 / %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>하중 데이터 열기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV 파일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>%1을(를) 읽을 수 없습니다: %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>파일</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>세션 열기…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>세션 저장…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>.aldic 파일을 pyALDIC에 연결…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>.aldic를 등록하여 세션 파일을 두 번 클릭하면 pyALDIC가 열리도록 합니다(현재 사용자만, 관리자 권한 불필요).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>종료</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>설정</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>언어</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>언어가 변경되었습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>언어가 %1(으)로 설정되었습니다. 모든 화면에 반영하려면 pyALDIC을 재시작하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>세션 저장</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC 세션</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>큼</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>결과를 포함할까요?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>이 세션에 계산된 결과를 포함하시겠습니까?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>결과를 포함하면(압축 전 약 %1) 다시 계산하지 않고 세션을 다시 열 수 있습니다. &apos;아니요&apos;를 선택하면 공유용으로 구성만 담긴 작은 파일을 저장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>세션 저장 중</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>세션 저장 실패</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>세션 열기</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>세션 불러오는 중</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>세션 열기 실패</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>세션 이미지 위치 지정</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>이 세션에 저장된 이미지 폴더를 찾을 수 없습니다:
%1

결과는 복원되었습니다. 배경 이미지를 표시하려면 현재 이미지가 들어 있는 폴더를 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>이미지 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>파일 연결 실패</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>.aldic 파일을 등록할 수 없습니다: </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>파일 연결</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>완료되었습니다. 이제 .aldic 파일을 두 번 클릭하면 pyALDIC가 열리고 해당 세션이 복원됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>프레임 %1에는 자체 관심 영역이 없습니다 — 계산에는 프레임 1의 관심 영역을 사용합니다. 프레임 1로 전환하여 편집하거나 마스크를 가져와 이 프레임 전용으로 지정하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>저장할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>관심 영역 마스크가 비어 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>관심 영역 마스크 저장</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>PNG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>마스크를 %1에 저장했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>반전할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>먼저 프레임 1에서 관심 영역을 정의하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  프레임 %1의 마스크를 가져왔습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>일괄 가져오기: 마스크 %n개를 불러왔습니다</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>메시 색상</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>메시 선 색상 선택을 위해 클릭</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>선 너비</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="37"/>
        <source>Subset Size</source>
        <translation>서브셋 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="43"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>IC-GN 서브셋 윈도우 크기(픽셀, 홀수)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="50"/>
        <source>Subset Step</source>
        <translation>서브셋 간격</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="53"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>노드 간격(픽셀, 2의 거듭제곱이어야 함)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Search Range</source>
        <translation>탐색 범위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Refine Inner Boundary</source>
        <translation>내부 경계 세분화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="79"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>내부 마스크 경계를 따라 메시를 국소적으로 세분화합니다
(관심 영역 내부의 구멍). 기포/공극 가장자리에 유용합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="83"/>
        <source>Refine Outer Boundary</source>
        <translation>외부 경계 세분화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>관심 영역 외부 경계를 따라 메시를 국소적으로 세분화합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>세분화 강도. 최소 요소 크기 = max(2, 서브셋 간격 / 2^레벨). 내부·외부 경계와 브러시로 칠한 영역에 모두 일괄 적용됩니다. 사용 가능한 레벨은 서브셋 크기와 간격에 따라 달라집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>세분화 레벨</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>FFT 탐색이 검출할 수 있는 프레임당 최대 변위(픽셀).
예상 프레임 간 움직임보다 충분히 크게 설정하세요.
증분 모드의 큰 회전 시 다음을 포함해야 합니다:
  반경 × sin(단계 각).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>각 시드점의 단점 NCC 탐색 초기 반폭(픽셀).
피크가 잘리면 이미지 절반 크기까지 재시도마다 2배씩 자동 확장합니다.
시드점 초기화에만 영향을 주며, 다른 노드는 F-aware 전파(노드별 탐색 없음)를 사용합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Initial Seed Search</source>
        <translation>초기 시드 탐색</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="218"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>약함</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="219"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>중간</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="220"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>강함</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="221"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>매우 강함</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="222"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>극강</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="228"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>최소 요소 크기 = %1 px  (서브셋 간격=%2, 레벨=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>물리 단위 사용</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>이미지 픽셀 1개의 물리 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="80"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>Pixel size</source>
        <translation>픽셀 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>획득 프레임 속도(속도장에 사용)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>프레임 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>변위: %1  속도: %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>변위: px  속도: px/fr</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>파이프라인 설정 구성 중…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>이미지 불러오는 중…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  %1 개 이미지 로드됨, shape=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  ROI 마스크: %1, %2 픽셀 (%3%)</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>실행 취소됨: 누락된 참조 프레임에 대해 프레임별 관심 영역을 정의하거나, 다음 실행 시 프레임 1의 마스크를 그대로 사용하도록 허용하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n 개 프레임에서 사용자 지정 ROI 마스크 사용</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>결과 수신: %n 프레임</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>DIC 분석 시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>분석 완료 (%1초)</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>사용자가 분석을 중지했습니다.</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>먼저 이미지를 불러온 후, 프레임 1에 관심 영역을 그리세요.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;누적 모드&lt;/b&gt; — 관심 영역은 프레임 1에만 필요합니다. 이후 프레임은 모두 직접 비교됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;증분, 매 프레임&lt;/b&gt; — 프레임 1에 관심 영역이 필요합니다. 이후 프레임으로 자동 전진 워프됩니다(프레임별 그리기 불필요).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;증분, %1 프레임마다&lt;/b&gt; — 다음 프레임에 관심 영역을 그리세요: &lt;b&gt;%2&lt;/b&gt;(기준 프레임 총 %3개).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;증분, 사용자 지정&lt;/b&gt; — 사용자 정의 기준 프레임이 설정되지 않았습니다. 프레임 1이 유일한 기준이 됩니다. 기준 프레임 입력란에 인덱스를 추가하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;증분, 사용자 지정&lt;/b&gt; — 다음 프레임에 관심 영역을 그리세요: &lt;b&gt;%1&lt;/b&gt;(기준 프레임 총 %2개).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>프레임 1에 관심 영역을 그리세요.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 추가</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>관심 영역에 도형을 추가합니다(다각형 / 사각형 / 원)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>잘라내기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>관심 영역에서 도형을 잘라냅니다(다각형 / 사각형 / 원)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 세분화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>브러시로 추가 메시 세분화 영역을 칠합니다
(프레임 1에서만 — 후속 프레임으로 자동 워프됨)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>세분화 브러시는 프레임 1에서만 사용할 수 있습니다. 프레임 1로 전환해 영역을 칠하세요. 후속 프레임으로 자동 워프됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>이미지 파일에서 마스크 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>여러 프레임의 마스크 파일을 일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>저장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>현재 마스크를 PNG 파일로 저장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>반전</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>관심 영역 마스크 반전</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>모든 관심 영역 마스크 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>반경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>칠하기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>브러시 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="256"/>
        <source>Circle (3-point)</source>
        <translation>원(3점)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="350"/>
        <source>Import Mask Image</source>
        <translation>마스크 이미지 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>DIC 분석 실행</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>취소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>현재 분석을 취소합니다. 이미 계산된 프레임은 유지되므로 부분 결과를 검토하거나 내보낼 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>변형률 창 열기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>별도의 후처리 창에서 변형률을 계산·시각화합니다. 완료된 실행의 변위 결과가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>진행률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>준비 완료</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>경과  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>남음  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>표시 필드</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>표시 기준</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>필드를 변형된 노드 위치에 그리거나 참조 프레임에서의 위치에 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="178"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="183"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>배경을 숨겼을 때 이미지를 대신할 채우기입니다. 내보낼 때 PNG와 TIFF는 투명도를 유지하며, 다른 형식은 흰색이 됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>시각화</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>색상표</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>오버레이 불투명도(0 = 투명, 100 = 불투명)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>로그</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>실행 전에 각 빨간 영역에 시드점을 하나 이상 배치하세요(빨강 = 시드점 필요).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  프레임 %2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="150"/>
        <source>DISPLACEMENT</source>
        <translation>변위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="161"/>
        <source>STRAIN</source>
        <translation>변형률</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>애니메이션 재생</translation>
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
        <translation>다음 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>재생 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>프레임 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>애니메이션 일시정지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>프레임 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>평면 피팅</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>FEM 절점</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Method</source>
        <translation>방법</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="102"/>
        <source>VSG (Virtual Strain Gauge) size is the diameter, in pixels, of the circular region around each mesh node used to fit a local displacement plane. Strain is then taken as the plane&apos;s slope.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).

Not used when Method = FEM nodal (there, mesh spacing itself sets the gauge size).</source>
        <translation>VSG(가상 변형률 게이지, Virtual Strain Gauge) 크기는 각 메시 노드 주위에서 국소 변위 평면을 피팅하는 데 사용되는 원형 영역의 지름(픽셀)입니다. 변형률은 이 평면의 기울기로 얻어집니다.

• VSG가 클수록 → 변형률이 매끄럽고 공간 해상도가 낮음.
• VSG가 작을수록 → 변형률이 날카롭지만 노이즈 증가.
• 경험 법칙: VSG ≥ 2 × 서브셋 스텝 + 1 (기본값: 41 px).

Method = FEM nodal일 때는 사용되지 않습니다(그 경우 메시 간격 자체가 게이지 크기를 결정).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>VSG 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>균일 메시에서 원형 VSG 윈도우 내 축당 메시 노드 수: 2 × floor(VSG 반경 / 노드 간격) + 1. 평면 피팅은 반경 내 모든 노드를 사용하며, 세분화된 메시에서는 이 수가 국소적으로 달라집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>ROI / 구멍 가장자리에서 VSG 창이 경계를 넘어 국소 평면 피팅이 한쪽으로 치우쳐 신뢰할 수 없게 되는 부분의 저신뢰도 변형률을 숨깁니다.

• 계수 × VSG 반경 = 잘라내는 경계 띠의 폭.
• 0.00 = 모든 노드 유지(잘라내기 없음).
• 0.70 = 권장(가장자리 오차가 급증하는 곳을 잘라냄).
• 1.00 = 가장 엄격(창이 가장자리에 닿는 모든 노드를 잘라냄).

Method = 평면 피팅 일 때만 적용됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>저신뢰도 가장자리 잘라내기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>끔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>약함 (σ = 0.5 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>중간 (σ = 1 × step)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>강함 (σ = 2 × step) ⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>계산 후 변형률장에 가우스 평활화를 적용합니다.
σ는 가우스 커널 너비, &apos;step&apos;은 DIC 노드 간격입니다.
  Light  (0.5 × step):  약함, 세부를 보존.
  Medium (1 × step):    균형, 노이즈 데이터에 권장.
  Strong (2 × step) ⚠: 강함, 실제 기울기를 흐릴 수 있음.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>변형률장 평활화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>미소 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>오일러 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>그린-라그랑주 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>변형률 종류</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>잘라냄: 노드 %1개 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>변형률 윈도우 ≈ %1×%2 노드</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ VSG 반경(%1 px) &lt; DIC 노드 간격(%2 px); 평면 피팅이 실패합니다. VSG ≥ %3 px로 설정하거나 Method를 FEM nodal로 전환하세요.</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>필드를 변형된 노드 위치에 그리거나 참조 프레임에서의 위치에 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>표시 기준</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="82"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="83"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="87"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>배경을 숨겼을 때 이미지를 대신할 채우기입니다. 내보낼 때 PNG와 TIFF는 투명도를 유지하며, 다른 형식은 흰색이 됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>배경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>색상 맵</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="122"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="134"/>
        <source>Opacity</source>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="143"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>잘라낸 가장자리 채우기 (표시 전용)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>가장자리가 잘린 변형률 띠를 신뢰할 수 있는 내부 노드에서 다시 보간합니다. 화면 표시와 내보낸 이미지/애니메이션에 영향을 줍니다. 내보낸 데이터 파일은 잘라낸 가장자리를 항상 NaN으로 유지합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>가장자리</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="155"/>
        <source>Strain Post-Processing</source>
        <translation>변형률 후처리</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit</source>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="204"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="214"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="272"/>
        <source>STRAIN PARAMETERS</source>
        <translation>변형률 매개변수</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="291"/>
        <source>Cancel</source>
        <translation>취소</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="295"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>실행 중인 변형률 계산을 취소합니다. 이전 변형률 결과는 유지됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>변위 및 변형률 결과를 NPZ / MAT / CSV / PNG로 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="340"/>
        <source>FIELD</source>
        <translation>표시 필드</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>VISUALIZATION</source>
        <translation>시각화</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="362"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="371"/>
        <source>LOG</source>
        <translation>로그</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="400"/>
        <source>Strain Field</source>
        <translation>변형률 장</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="402"/>
        <source>Analysis</source>
        <translation>분석</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="516"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>변형률 계산 실패: %1: %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="525"/>
        <location filename="../../gui/strain_window.py" line="587"/>
        <source>Strain computation complete.</source>
        <translation>변형률 계산 완료.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="536"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>변형률 창: 후처리할 변위 결과가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="567"/>
        <source>Cancelling…</source>
        <translation>취소 중…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="604"/>
        <source>Strain computation cancelled.</source>
        <translation>변형률 계산 취소됨.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="613"/>
        <source>Strain compute failed: %1</source>
        <translation>변형률 계산 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain Computation Failed</source>
        <translation>변형률 계산 실패</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="659"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 매개변수가 변경됨 — 「변형률 계산」을 클릭하세요</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="220"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="282"/>
        <source>Compute Strain</source>
        <translation>변형률 계산</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="304"/>
        <source>Export Results</source>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="547"/>
        <source>Starting…</source>
        <translation>시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="583"/>
        <source>Complete</source>
        <translation>완료</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>물리 단위 사용</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>단위: px/frame</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="51"/>
        <source>Incremental</source>
        <translation>증분형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="52"/>
        <source>Accumulative</source>
        <translation>누적형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>증분형: 각 프레임을 직전 기준 프레임과 비교합니다.
누적 변형이 큰 경우에 적합하며, 큰 회전에는 필수입니다.

누적형: 각 프레임을 1번 프레임과 비교합니다.
작고 단조로운 변형에만 정확합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>추적 모드</translation>
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
        <translation>Local DIC: 독립 서브셋 매칭(IC-GN). 빠르고
국소 특징을 보존합니다. 작은 변형이나 고품질
이미지에 적합합니다.

AL-DIC: 전역 FEM 정칙화를 갖춘 확장 라그랑주.
서브셋 간 변위 적합성을 강제합니다. 큰 변형,
노이즈 이미지, 변형률 정확도가 중요한 경우에 적합합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>솔버</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>매 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>N 프레임마다</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>사용자 지정 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>증분 추적 중 기준 프레임 갱신 시점.
매 프레임: 매 프레임마다 기준 리셋(단계 변위 최소,
큰 변형에 가장 견고).
N 프레임마다: N 프레임마다 리셋(속도-견고성 균형).
사용자 지정: 사용자 정의 기준 프레임 인덱스 목록.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>기준 프레임 갱신</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>N 프레임마다 기준 갱신</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>간격</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>기준 프레임으로 사용할 프레임 인덱스(0부터, 쉼표 구분)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>기준 프레임</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>이미지 폴더를 드롭하거나
찾아보기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>이미지 폴더 선택</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>미리보기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>(이미지 없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>이미지만</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>이미지 + 마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>마스크만</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>보기:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>알파:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>파랑</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>빨강</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>초록</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>노랑</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>마스크 색상:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>지정된 마스크 없음</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>프레임 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>이미지를 불러오지 못했습니다</translation>
    </message>
</context>
</TS>
