<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="44"/>
        <source>AL-DIC Iterations</source>
        <translation>AL-DIC 迭代次數</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="52"/>
        <source>Number of global refinement cycles for the AL-DIC solver.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>AL-DIC 求解器的全域精修迭代次數。
1 = 單次全域求解（最快），3 = 預設值，
5 次以上大多數情況下收益遞減。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="60"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>僅對 AL-DIC 求解器生效，Local DIC 會忽略。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="73"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>峰值被截斷時自動擴大 FFT 搜尋範圍</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="76"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>當 NCC 峰值觸及搜尋區域邊緣時，自動以更大的搜尋範圍重試（最大到影像一半尺寸，每次放大 2 倍，共 6 次重試）。

僅對 FFT 初始猜測模式有效。</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="135"/>
        <location filename="../../gui/widgets/mpl_chart.py" line="182"/>
        <source>Frame</source>
        <extracomment>Shading for frames a probe could not measure.</extracomment>
        <translation>影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="148"/>
        <source>Shaded frames: %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="184"/>
        <source>Distance along line (%1)</source>
        <translation>）</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="211"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <extracomment>Tool button label, tool token, and the probe kind it produces.</extracomment>
        <translation>點</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="212"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="213"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="214"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="215"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="218"/>
        <source>Click once to place a point probe.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="219"/>
        <source>Click twice: start and end of the gauge.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="220"/>
        <source>Click twice: opposite corners.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="221"/>
        <source>Click twice: centre, then the edge.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="223"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="231"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="232"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>稱</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="233"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="234"/>
        <source>Colour</source>
        <comment>Probe list column: colour swatch</comment>
        <translation>色</translation>
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
        <translation>除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="238"/>
        <source>Clear All</source>
        <translation>除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="239"/>
        <source>Compare:</source>
        <translation>：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="240"/>
        <source>Field:</source>
        <translation>：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="241"/>
        <source>Statistic:</source>
        <translation>：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="242"/>
        <source>Minimum valid fraction:</source>
        <translation>：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="244"/>
        <source>A frame is left blank when fewer than this fraction of the probe&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>。</translation>
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
        <translation>置</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="265"/>
        <source>Point probes</source>
        <translation>針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="266"/>
        <source>Line probes</source>
        <translation>針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="267"/>
        <source>Region probes</source>
        <translation>針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="294"/>
        <source>Value</source>
        <comment>Statistic: the sample itself, for a point probe</comment>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="295"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="296"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>數</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="297"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="298"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="299"/>
        <source>Standard deviation</source>
        <translation>差</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="300"/>
        <source>Valid fraction</source>
        <translation>例</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="301"/>
        <source>Engineering strain</source>
        <translation>變</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="302"/>
        <source>Crack opening</source>
        <translation>移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="332"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="391"/>
        <source>Clear All Probes</source>
        <translation>針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="392"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="432"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>點</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="433"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="434"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>區域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="473"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="484"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="502"/>
        <source>This statistic does not apply here.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="538"/>
        <location filename="../../gui/panels/analysis_tab.py" line="543"/>
        <source>Export Probe Data</source>
        <translation>料</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="539"/>
        <source>There is nothing to export yet.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="544"/>
        <source>CSV Files</source>
        <translation>案</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="544"/>
        <location filename="../../gui/panels/analysis_tab.py" line="570"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
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
        <translation>表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="568"/>
        <source>PNG Images</source>
        <translation>像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis_tab.py" line="569"/>
        <source>PDF Documents</source>
        <translation>件</translation>
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
        <translation>為 %n 影格匯入了感興趣區域</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="997"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>請先執行 DIC —— 目前沒有可後處理的位移結果。</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="1094"/>
        <source>pyALDIC has hit an error</source>
        <translation>誤</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1095"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1105"/>
        <source>Details were written to %1</source>
        <translation>1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1209"/>
        <source>Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="1223"/>
        <source>Compute kernels ready (%1 s).</source>
        <translation>。</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>根據每幀的資料範圍自動縮放顏色範圍</translation>
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
        <translation>所有幀都使用手動設定的最小/最大值</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="367"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批量匯入感興趣區域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="418"/>
        <source>Mask Folder:</source>
        <translation>掩模資料夾：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>(none)</source>
        <translation>（無）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="424"/>
        <source>Browse...</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="443"/>
        <source>Available Masks</source>
        <translation>可用掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="454"/>
        <source>Auto-Match by Name</source>
        <translation>按檔案名自動匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根據檔案名中的數字把掩模檔案匹配到對應幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="460"/>
        <source>Assign Sequential</source>
        <translation>順序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="462"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>從第 0 幀起按順序把掩模分配給各幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="470"/>
        <source>Frame Assignments</source>
        <translation>幀分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Frame</source>
        <translation>幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Image</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="473"/>
        <source>Mask</source>
        <translation>掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="484"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所選 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="486"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>將所選掩模與所選幀配對</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="535"/>
        <source>Select Mask Folder</source>
        <translation>選擇遮罩資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="589"/>
        <source>Failed to read mask file.</source>
        <translation>無法讀取遮罩檔案。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="593"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>尺寸不符：%1×%2（預期 %3×%4）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>%n 個遮罩尺寸不符，已停用。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="725"/>
        <source>Invalid assignment</source>
        <translation>無效的指派</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="727"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>一個影格只能對應一個遮罩。請選擇恰好一個遮罩，或選擇多個影格以將同一個遮罩套用到多個影格。</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1215"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1216"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到視口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1221"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
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
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1246"/>
        <source>Show Grid</source>
        <translation>顯示網格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1247"/>
        <source>Show/hide computational mesh grid</source>
        <translation>顯示/隱藏計算網格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1250"/>
        <source>Show Subset</source>
        <translation>顯示子集</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1251"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>懸停時顯示子集窗口（需要先開啟網格）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1505"/>
        <source>Placing Starting Points</source>
        <translation>正在放置種子點</translation>
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
        <translation>初始猜測</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="93"/>
        <source>Accumulative</source>
        <translation>累積式</translation>
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
        <translation>種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="112"/>
        <source>Previous frame</source>
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="114"/>
        <source>FFT every frame</source>
        <translation>每幀 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>每 %1 幀 FFT</translation>
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
        <translation>範圍</translation>
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
        <translation>自動</translation>
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
        <translation>欄位不透明度（0 = 透明，1 = 完全不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>All</source>
        <translation>全選</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>None</source>
        <translation>全不選</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>OUTPUT FOLDER</source>
        <translation>輸出資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select output folder…</source>
        <translation>選擇輸出資料夾…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Browse…</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="663"/>
        <source>Open Folder</source>
        <translation>開啟資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="671"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="675"/>
        <source>Enable physical units</source>
        <translation>啟用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="678"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸縮放位移值，並在色條標籤顯示物理單位。應變為無量綱，不受影響。</translation>
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
        <translation>幀率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="722"/>
        <source>Data</source>
        <translation>資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="723"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="724"/>
        <source>Animation</source>
        <translation>動畫</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="725"/>
        <source>Report</source>
        <translation>報告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="727"/>
        <source>Preview &amp; Colorbar</source>
        <translation>預覽與色條</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="758"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="760"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 歸檔 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="762"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="764"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐幀）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="767"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ：逐幀一個檔案（預設：合併為單個檔案）</translation>
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
        <translation>選擇：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="799"/>
        <source>STRAIN</source>
        <translation>應變</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="802"/>
        <source>Run Compute Strain first.</source>
        <translation>請先運行“計算應變”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 參數檔案（JSON）始終匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="835"/>
        <source>Export Data</source>
        <translation>匯出資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="856"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1005"/>
        <source>Export</source>
        <translation>匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="857"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1006"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1253"/>
        <source>Field</source>
        <translation>欄位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="858"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1007"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1355"/>
        <source>Colormap</source>
        <translation>色彩對映</translation>
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
        <translation>影像設定</translation>
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
        <translation>原始解析度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="900"/>
        <source>Cap the exported image&apos;s long edge (the larger of width/height; aspect ratio is kept).
Field detail is bounded by the mesh, so a smaller cap is near-lossless
but much smaller on disk and faster to encode. Lower = faster. &apos;Full resolution&apos; keeps the native size.</source>
        <translation>限制匯出影像的長邊（寬與高中較大的一個；保持長寬比）。
場的細節由網格密度決定，因此較小的上限幾乎無損，
但檔案更小、編碼更快。越低越快。「原始解析度」保持原生尺寸。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1054"/>
        <source>Resolution (long edge)</source>
        <translation>解析度（長邊）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="941"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1091"/>
        <source>Deformed frame</source>
        <translation>變形幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1092"/>
        <source>Reference frame</source>
        <translation>參考幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="946"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1096"/>
        <source>Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame&apos;s own photo.
Reference: drawn at the original node positions, over the first frame.</source>
        <translation>變形幀：欄位繪製在位移後的節點位置（參考位置 + 位移），疊加在每一幀自己的照片上。
參考幀：繪製在原始節點位置，疊加在第一幀上。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="954"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1102"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="957"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>取消勾選可僅匯出欄位，其後不含散斑影像。填充色於 Preview &amp; Colorbar 頁選擇。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1049"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution.</source>
        <translation>限制動畫的長邊（寬與高中較大的一個）。
越低越快、越小。強烈建議用於 GIF：其體積在原生解析度下會急劇膨脹。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="915"/>
        <source>JPEG quality (higher = larger file). Ignored for PNG/TIFF.</source>
        <translation>JPEG 品質（越高檔案越大）。對 PNG/TIFF 無效。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="918"/>
        <source>JPEG quality</source>
        <translation>JPEG 品質</translation>
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
        <translation>包含色條</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>Append a vertical colorbar strip to the right of each image.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>在每張影像右側添加一條垂直色條。
啟用自動範圍時，刻度標籤會依影格更新。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="963"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1111"/>
        <source>Render as</source>
        <translation>繪製為</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="981"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1127"/>
        <source>Cancel Export</source>
        <translation>取消匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="986"/>
        <source>Export Images</source>
        <translation>匯出影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1029"/>
        <source>ANIMATION SETTINGS</source>
        <translation>動畫設定</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1063"/>
        <source>FPS</source>
        <translation>影格率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1073"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>每 N 幀匯出一幀（1 = 每幀都匯出）。越大越快、越小，
但看起來更卡頓。播放時長保持不變（上方 FPS 為抽幀前的幀率）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1078"/>
        <source>Frame step</source>
        <translation>抽幀間隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1083"/>
        <source>Append a vertical colorbar strip to the right of each frame.
Tick labels update per frame when Auto range is enabled.</source>
        <translation>在每一影格右側添加一條垂直色條。
啟用自動範圍時，刻度標籤會依影格更新。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1105"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>取消勾選可僅匯出欄位，其後不含散斑影像。GIF 與 MP4 無法儲存透明度，透明填充將寫為白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1132"/>
        <source>Export Animation</source>
        <translation>匯出動畫</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1149"/>
        <source>CONTENT</source>
        <translation>內容</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1151"/>
        <source>Parameter summary table</source>
        <translation>參數摘要表</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1154"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>欄位統計（每影格 最小/最大/平均/標準差）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1157"/>
        <source>Sample field images</source>
        <translation>範例欄位影像</translation>
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
        <translation>影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1176"/>
        <source>FIELDS</source>
        <translation>欄位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1179"/>
        <source>Displacement:</source>
        <translation>位移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1194"/>
        <source>Strain:</source>
        <translation>應變：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1216"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>格式：HTML（自包含，可在任意瀏覽器中檢視）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1223"/>
        <source>Generate Report</source>
        <translation>產生報告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1245"/>
        <source>Open this tab to render a preview.</source>
        <translation>開啟此分頁以算繪預覽。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1261"/>
        <source>Frame</source>
        <translation>影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1273"/>
        <source>COLORBAR STYLE</source>
        <translation>色條樣式</translation>
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
        <translation>字級</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1292"/>
        <source>Font family</source>
        <translation>字型</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1300"/>
        <source>Bar thickness</source>
        <translation>色條粗細</translation>
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
        <translation>在匯出內容外圍加一圈空白邊框，寬度為長邊的比例（0 = 無）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1316"/>
        <source>Margin</source>
        <translation>邊距</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1322"/>
        <source>Margin color</source>
        <translation>邊距顏色</translation>
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
        <translation>關閉「顯示背景影像」時，用於填充原本顯示背景影像的區域。
PNG 與 TIFF 會保留透明度；JPEG、GIF 和 MP4 沒有 alpha 通道，將改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1340"/>
        <source>Hidden background</source>
        <translation>隱藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1342"/>
        <source>Refresh preview</source>
        <translation>重新整理預覽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1349"/>
        <source>FIELD APPEARANCE</source>
        <translation>欄位外觀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1360"/>
        <source>Range</source>
        <translation>範圍</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1384"/>
        <source>Apply to all fields</source>
        <translation>套用到所有欄位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1386"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>將該欄位的 colormap、不透明度和自動範圍套用到所有已啟用欄位（每個欄位保留各自的 min/max）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1509"/>
        <source>Preview failed: </source>
        <translation>預覽失敗：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1524"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>在 Images 頁啟用一個欄位以進行預覽。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1537"/>
        <source>No data for this field/frame.</source>
        <translation>該欄位/影格沒有資料。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1602"/>
        <source>FRAME RANGE</source>
        <translation>影格範圍</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1605"/>
        <source>All frames</source>
        <translation>所有影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1611"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>從</translation>
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
        <translation>選擇輸出資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1709"/>
        <source>Exported %1 files → %2</source>
        <translation>已匯出 %1 個檔案 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1718"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1848"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1937"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1979"/>
        <source>Error: %1</source>
        <translation>錯誤：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1743"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1876"/>
        <source>Starting…</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1821"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1899"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>正在繪製 %1 (%2/%3)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1827"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="1905"/>
        <source>Frame %1/%2</source>
        <translation>影格 %1/%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1837"/>
        <source>Exported %1 images → %2</source>
        <translation>已匯出 %1 張影像 → %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1915"/>
        <source>No animation was written. See the log for details.</source>
        <translation>。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="1972"/>
        <source>Report saved → %1</source>
        <translation>報告已儲存 → %1</translation>
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
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>播放動畫</translation>
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
        <translation>下一幀</translation>
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
        <translation>第 0/0 幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>暫停動畫</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>影格 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1061"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>請先載入影像，再繪製感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1080"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>三個點幾乎共線 — 請在圓周上分散地選取三個點。</translation>
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
        <translation>檔名</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>區域</translation>
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
        <translation>編輯</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="245"/>
        <source>Need</source>
        <translation>待繪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="362"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>為 %n 影格匯入感興趣區域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="378"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>清除感興趣區域（%1 影格已有區域）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="382"/>
        <source>Clear Region of Interest</source>
        <translation>清除感興趣區域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="393"/>
        <source>Delete %n image(s)</source>
        <translation>刪除 %n 張影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="496"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="497"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="506"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>已選擇 %1 個檔案用於 %2 影格 — 數量必須相符</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="80"/>
        <source>Starting Points</source>
        <translation>種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="84"/>
        <source>Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.

Best for:
• Large inter-frame displacement (&gt; 50 px)
• Discontinuous fields (cracks, shear bands)
• Scenarios where FFT picks wrong peaks

Auto-placed per region when you draw or edit an ROI.</source>
        <translation>放置若干種子點；pyALDIC 在每個點上運行單點 NCC 引導，然後沿網格鄰居傳播位移場。

最適合：
• 大幀間位移（&gt; 50 px）
• 不連續場（裂紋、剪切帶）
• FFT 容易選錯峰的場景

繪製或編輯 ROI 時會為每個區域自動放置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="100"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="281"/>
        <source>Place Starting Points</source>
        <translation>放置種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="103"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>在畫布上進入放置模式。左鍵添加、右鍵刪除，按 Esc 或再次點擊離開。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="107"/>
        <source>Auto-place</source>
        <translation>自動放置</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="109"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>在每個空區域填入 NCC 最高的節點。已有種子點會保留。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="113"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="115"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>移除所有種子點。比逐個右鍵刪除快。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="292"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 區域就緒</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="141"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT（互相關）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="145"/>
        <source>Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.

Best for:
• Small-to-moderate smooth motion
• Well-textured speckle
• No special user setup needed

Cost grows with the search radius, so very large displacements become slow.</source>
        <translation>全網格歸一化互相關。在搜尋半徑內穩健；峰值被截斷時搜尋自動擴展。

最適合：
• 小到中等的平滑運動
• 紋理良好的散斑
• 不需要用戶額外設定

計算成本隨搜尋半徑增長，極大位移會變慢。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="163"/>
        <source>Every</source>
        <translation>每</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="165"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>每 N 幀運行一次 FFT。N = 1 表示每幀都做 FFT（最安全但最慢）。N &gt; 1 在兩次重置之間使用熱啟動，將誤差傳播限制在 N 幀內。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="175"/>
        <source>(N=1 = every frame)</source>
        <translation>（N=1 即每幀）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="184"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>僅在參考幀更新時（只對增量模式）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="187"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>參考幀變化時運行 FFT；每段內使用熱啟動。是增量模式的典型預設值。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="198"/>
        <source>Previous frame</source>
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="202"/>
        <source>Use the previous frame&apos;s converged displacement as the initial guess. No cross-correlation runs.

Best for:
• Very small inter-frame motion (a few pixels)
• Fastest option when motion is smooth

Errors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.</source>
        <translation>使用前一幀收斂的位移作為初始猜測。不運行任何互相關。

最適合：
• 非常小的幀間運動（幾像素）
• 運動平滑時速度最快

長序列中誤差會累積。資料有噪聲或運動較大時請選 FFT 或種子點。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="280"/>
        <source>Placing... (click to exit)</source>
        <translation>放置中…（再次點擊離開）</translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="178"/>
        <source>IMAGES</source>
        <translation>影像</translation>
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
        <translation>按檔案名中的數字排序：image1, image2, …, image10
預設（不勾選）：字典序 — 適合已補零的檔案名</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="261"/>
        <source>WORKFLOW TYPE</source>
        <translation>工作流類型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="274"/>
        <source>INITIAL GUESS</source>
        <translation>初始猜測</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="281"/>
        <source>REGION OF INTEREST</source>
        <translation>感興趣區域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="291"/>
        <source>PARAMETERS</source>
        <translation>參數</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="297"/>
        <source>ADVANCED</source>
        <translation>高級</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;File</source>
        <translation>檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <source>Open Session…</source>
        <translation>開啟會話…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="208"/>
        <source>Save Session…</source>
        <translation>儲存會話…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="217"/>
        <source>Associate .aldic files with pyALDIC…</source>
        <translation>將 .aldic 檔案關聯到 pyALDIC…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Register .aldic so double-clicking a session file opens pyALDIC (current user only, no admin rights needed).</source>
        <translation>註冊 .aldic，讓雙擊工作階段檔案即可開啟 pyALDIC（僅目前使用者，無需系統管理員權限）。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="225"/>
        <source>Quit</source>
        <translation>離開</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="233"/>
        <source>&amp;Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="234"/>
        <source>Language</source>
        <translation>語言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <source>Language changed</source>
        <translation>語言已切換</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>語言已切換至 %1。請重啟 pyALDIC 以讓所有界面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Save Session</source>
        <translation>儲存會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="282"/>
        <location filename="../../gui/app.py" line="331"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC 會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <location filename="../../gui/app.py" line="332"/>
        <location filename="../../gui/app.py" line="796"/>
        <source>All Files</source>
        <translation>全部檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="296"/>
        <source>large</source>
        <translation>較大</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Include Results?</source>
        <translation>包含結果？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Include the computed results in this session?</source>
        <translation>在此工作階段中包含已計算的結果嗎？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="304"/>
        <source>Including results (about %1 uncompressed) lets you reopen the session without recomputing. Choose No to save a small configuration-only file for sharing.</source>
        <translation>包含結果（未壓縮約 %1）可讓你下次直接開啟工作階段而無需重新計算。選擇「否」則只儲存一個小的僅設定檔案，便於分享。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="317"/>
        <source>Saving Session</source>
        <translation>正在儲存工作階段</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>Save Session Failed</source>
        <translation>儲存會話失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Open Session</source>
        <translation>開啟會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="344"/>
        <source>Loading Session</source>
        <translation>正在載入工作階段</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <location filename="../../gui/app.py" line="382"/>
        <source>Open Session Failed</source>
        <translation>開啟會話失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="356"/>
        <source>Locate Session Images</source>
        <translation>定位工作階段圖片</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="359"/>
        <source>The image folder saved with this session was not found:
%1

Results were restored. To show the background images, select the folder that now contains them.</source>
        <translation>找不到此工作階段儲存的圖片資料夾:
%1

結果已還原。若要顯示背景圖片,請選擇現在包含這些圖片的資料夾。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>Select Image Folder</source>
        <translation>選擇影像資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="443"/>
        <source>File Association Failed</source>
        <translation>檔案關聯失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="444"/>
        <source>Could not register .aldic files: </source>
        <translation>無法註冊 .aldic 檔案：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="447"/>
        <source>File Association</source>
        <translation>檔案關聯</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="448"/>
        <source>Done. Double-clicking a .aldic file will now open pyALDIC and restore that session.</source>
        <translation>完成。現在雙擊 .aldic 檔案即可開啟 pyALDIC 並還原該工作階段。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="763"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>幀 %1 沒有自己的感興趣區域 — 計算時使用幀 1 的感興趣區域。請切換到幀 1 編輯，或匯入遮罩為此幀單獨指定。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="774"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>沒有可儲存的感興趣區域 — 請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="781"/>
        <source>Region of Interest mask is empty.</source>
        <translation>感興趣區域遮罩為空。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="793"/>
        <source>Save Region of Interest Mask</source>
        <translation>罩</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="795"/>
        <source>PNG Images</source>
        <translation>像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="804"/>
        <source>Mask saved to %1</source>
        <translation>遮罩已儲存至 %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="813"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>沒有可反選的感興趣區域 — 請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="843"/>
        <location filename="../../gui/app.py" line="896"/>
        <source>Load images first.</source>
        <translation>請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="854"/>
        <source>Define a Region of Interest on frame 1 first.</source>
        <translation>請先在幀 1 上定義感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="945"/>
        <source>  Imported mask for frame %1</source>
        <translation>  已匯入影格 %1 的遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="951"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>批次匯入：已載入 %n 個遮罩</translation>
    </message>
</context>
<context>
    <name>MeshAppearanceWidget</name>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="38"/>
        <source>Mesh color</source>
        <translation>網格顏色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="41"/>
        <source>Click to choose mesh line color</source>
        <translation>點擊選擇網格線顏色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mesh_appearance_widget.py" line="50"/>
        <source>Line width</source>
        <translation>線寬</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="37"/>
        <source>Subset Size</source>
        <translation>子集尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="43"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>IC-GN 子集窗口尺寸（像素，奇數）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="50"/>
        <source>Subset Step</source>
        <translation>子集步長</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="53"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>節點間距（像素，必須是 2 的冪）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Search Range</source>
        <translation>搜尋範圍</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Refine Inner Boundary</source>
        <translation>加密內部邊界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="79"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>沿內部掩模邊界局部加密網格
（感興趣區域內部的孔洞）。適合氣泡 / 空洞邊緣。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="83"/>
        <source>Refine Outer Boundary</source>
        <translation>加密外部邊界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>沿感興趣區域的外部邊界局部加密網格。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>加密強度。最小單元尺寸 = max(2, 子集步長 / 2^級別)。對內部邊界、外部邊界和畫筆加密區域統一生效。可用級別取決於子集尺寸和步長。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>加密級別</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Maximum per-frame displacement the FFT search can detect (pixels).
Set comfortably larger than the expected inter-frame motion.
For large rotations in incremental mode, this must cover
  radius × sin(per-step angle).</source>
        <translation>FFT 搜尋可檢測到的每幀最大位移（像素）。
設定值應略大於預期的幀間運動。
對於增量模式下的大旋轉，該值必須覆蓋
  半徑 × sin(單步角度)。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>每個種子點處單點 NCC 搜尋的初始半寬（像素）。
若峰值被截斷，每次重試自動放大 2 倍，最大到影像一半尺寸。
僅影響種子點引導；其他節點使用 F-aware 傳播（無需逐節點搜尋）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Initial Seed Search</source>
        <translation>初始種子搜尋</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="218"/>
        <source>Light</source>
        <comment>Mesh refinement severity</comment>
        <translation>輕度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="219"/>
        <source>Medium</source>
        <comment>Mesh refinement severity</comment>
        <translation>中等</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="220"/>
        <source>Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>強</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="221"/>
        <source>Extra Heavy</source>
        <comment>Mesh refinement severity</comment>
        <translation>超強</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="222"/>
        <source>Ultra</source>
        <comment>Mesh refinement severity</comment>
        <translation>極限</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="228"/>
        <source>%1 (L%2)</source>
        <translation>%1 (L%2)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="250"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>最小單元尺寸 = %1 px  (子集步長=%2, 級別=%3)</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="54"/>
        <source>Use physical units</source>
        <translation>使用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="65"/>
        <source>Physical size of one image pixel</source>
        <translation>單個影像像素對應的物理尺寸</translation>
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
        <translation>採集幀率（用於速度場）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="94"/>
        <source>Frame rate</source>
        <translation>影格率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="162"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>位移：%1  速度：%2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="167"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/幀</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="394"/>
        <source>Building pipeline configuration...</source>
        <translation>正在構建流水線配置…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="552"/>
        <source>Loading images...</source>
        <translation>正在載入影像…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="565"/>
        <source>  Loaded %1 images, shape=%2</source>
        <translation>  已載入 %1 張影像，尺寸=%2</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="578"/>
        <source>  ROI mask: %1, %2 pixels (%3%)</source>
        <translation>  感興趣區域遮罩：%1，%2 像素（%3%）</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="604"/>
        <source>Run cancelled: define per-frame Regions of Interest for the missing reference frames or accept the inherited frame-1 mask in the next run.</source>
        <translation>已取消執行：請為缺失的參考影格定義逐影格感興趣區域，或在下次執行時接受繼承自第 1 影格的遮罩。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="625"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>  %n 影格使用自訂感興趣區域遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="755"/>
        <source>Results received: %n frame(s)</source>
        <translation>已收到結果：%n 影格</translation>
    </message>
</context>
<context>
    <name>PipelineWorker</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>Starting DIC analysis...</source>
        <translation>開始 DIC 分析…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="257"/>
        <source>Analysis complete in %1s</source>
        <translation>分析完成，耗時 %1 秒</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="250"/>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>Analysis stopped by user.</source>
        <translation>使用者已停止分析。</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="62"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>請先載入影像，再在第 1 幀上繪製感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="69"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累積模式&lt;/b&gt; — 只有第 1 幀需要感興趣區域。後續幀都直接與其比較。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="79"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;增量模式，每幀&lt;/b&gt; — 第 1 幀需要感興趣區域。系統會自動將其扭曲到每個後續幀（無需逐幀繪製）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，每 %1 幀&lt;/b&gt; — 請在以下幀繪製感興趣區域：&lt;b&gt;%2&lt;/b&gt;（共 %3 個參考幀）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="110"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;增量模式，自定義&lt;/b&gt; — 未設定自定義參考幀。僅第 1 幀為參考；請在參考幀清單中添加更多索引。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="120"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，自定義&lt;/b&gt; — 請在以下幀繪製感興趣區域：&lt;b&gt;%1&lt;/b&gt;（共 %2 個參考幀）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="128"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>請在第 1 幀繪製感興趣區域。</translation>
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
        <translation>向感興趣區域添加形狀（多邊形 / 矩形 / 圓形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>從感興趣區域裁剪形狀（多邊形 / 矩形 / 圓形）</translation>
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
        <translation>用畫筆繪製額外的網格加密區域
（僅在第 1 幀可用 — 網格點會自動扭曲到後續幀）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>加密畫筆僅在第 1 幀可用。切換到第 1 幀後可繪製加密區域；系統會自動將其扭曲到後續幀。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>匯入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>從影像檔案匯入掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>批量匯入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>批量匯入多幀的掩模檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>儲存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>將當前掩模儲存為 PNG 檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>反選</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>反轉感興趣區域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>清除所有感興趣區域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>半徑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>繪製</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>擦除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>清除畫筆</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="256"/>
        <source>Circle (3-point)</source>
        <translation>圓（三點）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="350"/>
        <source>Import Mask Image</source>
        <translation>像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="353"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="59"/>
        <source>Run DIC Analysis</source>
        <translation>運行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="72"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="77"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>取消當前分析。已計算的影格會被保留，你可以檢視或匯出這部分結果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="85"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Open Strain Window</source>
        <translation>開啟應變窗口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="96"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在獨立的後處理窗口中計算並可視化應變。需先完成一次運行以獲得位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="105"/>
        <source>PROGRESS</source>
        <translation>進度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="114"/>
        <location filename="../../gui/panels/right_sidebar.py" line="392"/>
        <source>Ready</source>
        <translation>就緒</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <location filename="../../gui/panels/right_sidebar.py" line="394"/>
        <location filename="../../gui/panels/right_sidebar.py" line="485"/>
        <source>ELAPSED  %1</source>
        <translation>已用  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="127"/>
        <location filename="../../gui/panels/right_sidebar.py" line="396"/>
        <location filename="../../gui/panels/right_sidebar.py" line="493"/>
        <location filename="../../gui/panels/right_sidebar.py" line="497"/>
        <source>REMAINING  %1</source>
        <translation>剩餘  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="135"/>
        <source>FIELD</source>
        <translation>場變量</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <source>Show on</source>
        <translation>顯示於</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="149"/>
        <source>Deformed frame</source>
        <translation>變形幀</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="150"/>
        <source>Reference frame</source>
        <translation>參考幀</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>將欄位繪製在變形後的節點位置，或繪製在其於參考幀中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="163"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾選可僅顯示欄位，其後不顯示散斑影像。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="174"/>
        <source>Hidden background</source>
        <translation>隱藏背景填充</translation>
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
        <translation>背景隱藏時用什麼替代影像。匯出時 PNG 與 TIFF 會保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="193"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>Colormap</source>
        <translation>色圖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="217"/>
        <source>Opacity</source>
        <translation>透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="224"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>疊加圖透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="236"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="242"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="361"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>運行前請在每個紅色區域放置至少一個種子點（紅色 = 需要種子點）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="468"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  第 %2 幀</translation>
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
        <translation>應變</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>播放動畫</translation>
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
        <translation>下一幀</translation>
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
        <translation>第 0/0 幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>暫停動畫</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>影格 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="85"/>
        <source>Plane fitting</source>
        <translation>平面擬合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>FEM nodal</source>
        <translation>有限元節點</translation>
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
        <translation>VSG（虛擬應變計，Virtual Strain Gauge）尺寸指圍繞每個網格節點、用於擬合局部位移平面的圓形區域的直徑（像素）。應變由該平面的斜率給出。

• VSG 越大 → 應變越平滑，空間解析度越低。
• VSG 越小 → 應變越銳利，但雜訊越大。
• 經驗法則：VSG ≥ 2 × 子集步長 + 1（預設：41 px）。

方法選擇 FEM nodal 時不使用此參數（此時由網格間距決定虛擬應變計尺寸）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>VSG size</source>
        <translation>VSG 尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Number of mesh nodes per axis inside the circular VSG window on a uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane fit uses every node within the radius; on a refined mesh the count varies locally.</source>
        <translation>均勻網格下圓形 VSG 視窗內每個軸向的網格節點數：2 × floor(VSG 半徑 / 節點間距) + 1。平面擬合使用半徑內的所有節點；在加密網格上該數量會局部變化。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="162"/>
        <source>Hides low-confidence strain at ROI / hole edges, where the VSG window crosses the boundary and the local plane fit becomes one-sided and unreliable.

• Coefficient × VSG radius = width of the trimmed boundary band.
• 0.00 = keep every node (no trimming).
• 0.70 = recommended (trims where edge error rises sharply).
• 1.00 = strictest (trim any node whose window touches the edge).

Only applies when Method = Plane fitting.</source>
        <translation>在 ROI / 孔洞邊緣隱藏低可信度的應變：那裡 VSG 視窗跨越邊界，局部平面擬合變成單邊、不可靠。

• 係數 × VSG 半徑 = 裁剪邊界帶的寬度。
• 0.00 = 保留所有節點（不裁剪）。
• 0.70 = 推薦（裁掉誤差明顯上升的區域）。
• 1.00 = 最嚴格（視窗一旦觸及邊界即裁剪）。

僅在 方法 = 平面擬合 時生效。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Trim low-confidence edges</source>
        <translation>裁剪低可信度邊緣</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="201"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>關閉</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Light (σ = 0.5 × step)</source>
        <translation>輕度（σ = 0.5 × step）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="203"/>
        <source>Medium (σ = 1 × step)</source>
        <translation>中等（σ = 1 × step）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="204"/>
        <source>Strong (σ = 2 × step) ⚠</source>
        <translation>強（σ = 2 × step）⚠</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Gaussian smoothing of the strain field after computation.
σ is the Gaussian kernel width; &apos;step&apos; = DIC node spacing.
  Light  (0.5 × step):  subtle, preserves fine features.
  Medium (1 × step):    balanced, recommended for noisy data.
  Strong (2 × step) ⚠:  aggressive, may blur real gradients.</source>
        <translation>計算後對應變場做高斯平滑。
σ 為高斯核寬度；“step” 為 DIC 節點間距。
  Light（0.5 × step）：輕度平滑，保留細節。
  Medium（1 × step）：平衡選擇，推薦用於噪聲資料。
  Strong（2 × step）⚠：強平滑，可能模糊真實梯度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="216"/>
        <source>Strain field smoothing</source>
        <translation>應變場平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="221"/>
        <source>Infinitesimal</source>
        <translation>無窮小應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="222"/>
        <source>Eulerian</source>
        <translation>歐拉應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="223"/>
        <source>Green-Lagrangian</source>
        <translation>格林-拉格朗日應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="225"/>
        <source>Strain type</source>
        <translation>應變類型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="287"/>
        <source>Trimmed: %1 nodes (%2%)</source>
        <translation>已裁剪：%1 個節點 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="363"/>
        <source>Strain window ≈ %1×%2 nodes</source>
        <translation>應變視窗 ≈ %1×%2 節點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="370"/>
        <source>⚠ VSG radius (%1 px) &lt; DIC node spacing (%2 px); plane fit will fail. Use VSG ≥ %3 px or switch Method to FEM nodal.</source>
        <translation>⚠ VSG 半徑（%1 px）&lt; DIC 節點間距（%2 px）；平面擬合將失敗。請將 VSG ≥ %3 px 或將方法切換為 FEM nodal。</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="65"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in al_dic.core.colormaps, so every chooser offers the same options.</extracomment>
        <translation>變形幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="66"/>
        <source>Reference frame</source>
        <translation>參考幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>將欄位繪製在變形後的節點位置，或繪製在其於參考幀中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="71"/>
        <source>Show on</source>
        <translation>顯示於</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="76"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾選可僅顯示欄位，其後不顯示散斑影像。</translation>
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
        <translation>背景隱藏時用什麼替代影像。匯出時 PNG 與 TIFF 會保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="79"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="91"/>
        <source>Hidden background</source>
        <translation>隱藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Colormap</source>
        <translation>色彩對映</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="102"/>
        <source>Range</source>
        <translation>範圍</translation>
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
        <translation>填充修剪的邊緣（僅顯示）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Re-interpolate the edge-trimmed strain band from reliable interior nodes. Affects the on-screen view and exported images/animations; exported data files always keep the trimmed edge as NaN.</source>
        <translation>從可靠的內部節點重新插值被邊緣修剪的應變帶。影響螢幕顯示與匯出的圖片/動畫；匯出的資料檔案始終將修剪的邊緣保留為 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="152"/>
        <source>Edges</source>
        <translation>邊緣</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="154"/>
        <source>Strain Post-Processing</source>
        <translation>應變後處理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="202"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="203"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到視口</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="209"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="213"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="271"/>
        <source>STRAIN PARAMETERS</source>
        <translation>應變參數</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="290"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="294"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>取消正在進行的應變計算。保留先前的應變結果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="306"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>將位移和應變結果匯出為 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="339"/>
        <source>FIELD</source>
        <translation>場變量</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="348"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="361"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="370"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="399"/>
        <source>Strain Field</source>
        <translation>場</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="401"/>
        <source>Analysis</source>
        <translation>析</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="476"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>應變計算失敗：%1：%2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="484"/>
        <location filename="../../gui/strain_window.py" line="543"/>
        <source>Strain computation complete.</source>
        <translation>應變計算完成。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="495"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>應變視窗：沒有可後處理的位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="526"/>
        <source>Cancelling…</source>
        <translation>正在取消…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="560"/>
        <source>Strain computation cancelled.</source>
        <translation>應變計算已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="569"/>
        <source>Strain compute failed: %1</source>
        <translation>應變計算失敗：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="576"/>
        <source>Strain Computation Failed</source>
        <translation>應變計算失敗</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="615"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 參數已變更 — 請點擊“計算應變”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="219"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="281"/>
        <source>Compute Strain</source>
        <translation>計算應變</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="303"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="506"/>
        <source>Starting…</source>
        <translation>啟動中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="539"/>
        <source>Complete</source>
        <translation>完成</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>使用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>單位：px/幀</translation>
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
        <translation>累積式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="57"/>
        <source>Incremental: each frame is compared to the previous reference frame.
Suitable for large accumulated deformation, required for large rotations.

Accumulative: every frame is compared to frame 1.
Accurate for small, monotonic deformation only.</source>
        <translation>增量式：每幀與前一個參考幀比較。
適用於大量累積變形，大旋轉場景必須使用。

累積式：每幀都與第 1 幀比較。
僅適用於小的、單調的變形。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Tracking Mode</source>
        <translation>追蹤模式</translation>
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
        <translation>Local DIC：獨立子集匹配（IC-GN）。速度快，
保留局部銳利特徵。適合小變形
或高質量影像。

AL-DIC：全局 FEM 正則化的增廣拉格朗日方法。
強制子集間的位移相容性。適合大變形、
噪聲影像，或對應變精度要求高的場景。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>Solver</source>
        <translation>求解器</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Every Frame</source>
        <translation>每幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>Every N Frames</source>
        <translation>每 N 幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="106"/>
        <source>Custom Frames</source>
        <translation>自定義幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="109"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>增量追蹤中參考幀的刷新策略。
每幀：每幀都更新參考（單步位移最小，
對大變形最穩健）。
每 N 幀：每 N 幀更新一次（速度與穩健性的折中）。
自定義：由用戶指定參考幀索引清單。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Reference Update</source>
        <translation>參考幀更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="128"/>
        <source>Update reference every N frames</source>
        <translation>每 N 幀更新一次參考幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="130"/>
        <source>Interval</source>
        <translation>間隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>用作參考幀的幀索引清單（0 為起始），用逗號分隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="143"/>
        <source>Reference Frames</source>
        <translation>參考幀清單</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="120"/>
        <source>Drop image folder
or Browse</source>
        <translation>拖入影像資料夾
或點擊瀏覽</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="130"/>
        <source>Select Image Folder</source>
        <translation>選擇影像資料夾</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="132"/>
        <source>Preview</source>
        <translation>預覽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="136"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="262"/>
        <source>(no image)</source>
        <translation>（無影像）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="156"/>
        <source>Image only</source>
        <translation>僅影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image + Mask</source>
        <translation>影像 + 遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Mask only</source>
        <translation>僅遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="161"/>
        <source>View:</source>
        <translation>檢視：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="167"/>
        <source>Alpha:</source>
        <translation>透明度：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="175"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>藍色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="179"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>紅色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="183"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>綠色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="187"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>黃色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="191"/>
        <source>Mask color:</source>
        <translation>遮罩顏色：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="268"/>
        <source>No mask assigned</source>
        <translation>未指派遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="274"/>
        <source>Frame %1 — %2</source>
        <translation>影格 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="289"/>
        <source>Failed to load image</source>
        <translation>無法載入影像</translation>
    </message>
</context>
</TS>
