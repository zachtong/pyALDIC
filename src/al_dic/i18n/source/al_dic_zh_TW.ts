<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="39"/>
        <source>ADMM Iterations</source>
        <translation>ADMM 迭代次數</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="50"/>
        <source>Number of ADMM alternating minimization cycles for AL-DIC.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>AL-DIC 的 ADMM 交替最小化循環次數。
1 = 單次全局迭代（最快），3 = 預設值，
5 及以上對大多數場景收益遞減。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="56"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>僅對 AL-DIC 求解器生效，Local DIC 會忽略。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="69"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>峰值被截斷時自動擴大 FFT 搜尋範圍</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="75"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>當 NCC 峰值觸及搜尋區域邊緣時，自動以更大的搜尋範圍重試（最大到影像一半尺寸，每次放大 2 倍，共 6 次重試）。

僅對 FFT 初始猜測模式有效。</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="45"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批量匯入感興趣區域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="64"/>
        <source>Mask Folder:</source>
        <translation>掩模資料夾：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="65"/>
        <source>(none)</source>
        <translation>（無）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="70"/>
        <source>Browse...</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="80"/>
        <source>Available Masks</source>
        <translation>可用掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="87"/>
        <source>Auto-Match by Name</source>
        <translation>按檔案名自動匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="89"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根據檔案名中的數字把掩模檔案匹配到對應幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="93"/>
        <source>Assign Sequential</source>
        <translation>順序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="95"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>從第 0 幀起按順序把掩模分配給各幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="103"/>
        <source>Frame Assignments</source>
        <translation>幀分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Frame</source>
        <translation>幀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Image</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Mask</source>
        <translation>掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="116"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所選 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="118"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>將所選掩模與所選幀配對</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="122"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1107"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1108"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到視口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1113"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1114"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1118"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1124"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1125"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1138"/>
        <source>Show Grid</source>
        <translation>顯示網格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1139"/>
        <source>Show/hide computational mesh grid</source>
        <translation>顯示/隱藏計算網格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1142"/>
        <source>Show Subset</source>
        <translation>顯示子集</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1143"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>懸停時顯示子集窗口（需要先開啟網格）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1397"/>
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
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="102"/>
        <source>Local DIC</source>
        <translation>Local DIC</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="103"/>
        <source>ADMM (%1 iter)</source>
        <translation>ADMM（%1 次迭代）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="111"/>
        <source>Starting Points</source>
        <translation>種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="113"/>
        <source>Previous frame</source>
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="115"/>
        <source>FFT every frame</source>
        <translation>每幀 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>每 %1 幀 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="123"/>
        <source>FFT</source>
        <translation>FFT</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="29"/>
        <source>Range</source>
        <translation>範圍</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="30"/>
        <source>Auto</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="41"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="51"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="469"/>
        <source>All</source>
        <translation>全選</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="471"/>
        <source>None</source>
        <translation>全不選</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="500"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="520"/>
        <source>OUTPUT FOLDER</source>
        <translation>輸出資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="528"/>
        <source>Select output folder…</source>
        <translation>選擇輸出資料夾…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="532"/>
        <source>Browse…</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="537"/>
        <source>Open Folder</source>
        <translation>開啟資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="545"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="549"/>
        <source>Enable physical units</source>
        <translation>啟用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="554"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸縮放位移值，並在色條標籤顯示物理單位。應變為無量綱，不受影響。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="569"/>
        <source>/ pixel</source>
        <translation>/ 像素</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="571"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="580"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="582"/>
        <source>Frame rate</source>
        <translation>幀率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="590"/>
        <source>Data</source>
        <translation>資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="592"/>
        <source>Animation</source>
        <translation>動畫</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>Report</source>
        <translation>報告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="613"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="615"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 歸檔 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="617"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="619"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐幀）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ：逐幀一個檔案（預設：合併為單個檔案）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="630"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="639"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="666"/>
        <source>Select:</source>
        <translation>選擇：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>STRAIN</source>
        <translation>應變</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="657"/>
        <source>Run Compute Strain first.</source>
        <translation>請先運行“計算應變”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="684"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 參數檔案（JSON）始終匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="690"/>
        <source>Export Data</source>
        <translation>匯出資料</translation>
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
</context>
<context>
    <name>ImageList</name>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="186"/>
        <location filename="../../gui/widgets/image_list.py" line="244"/>
        <source>Add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="238"/>
        <source>Edit</source>
        <translation>編輯</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="241"/>
        <source>Need</source>
        <translation>待繪</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="111"/>
        <source>Starting Points</source>
        <translation>種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="131"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="313"/>
        <source>Place Starting Points</source>
        <translation>放置種子點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="136"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>在畫布上進入放置模式。左鍵添加、右鍵刪除，按 Esc 或再次點擊離開。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="138"/>
        <source>Auto-place</source>
        <translation>自動放置</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="142"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>在每個空區域填入 NCC 最高的節點。已有種子點會保留。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="144"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="148"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>移除所有種子點。比逐個右鍵刪除快。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="153"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="323"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 區域就緒</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="172"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT（互相關）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="183"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="194"/>
        <source>Every</source>
        <translation>每</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="199"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>每 N 幀運行一次 FFT。N = 1 表示每幀都做 FFT（最安全但最慢）。N &gt; 1 在兩次重置之間使用熱啟動，將誤差傳播限制在 N 幀內。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="206"/>
        <source>(N=1 = every frame)</source>
        <translation>（N=1 即每幀）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="216"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>僅在參考幀更新時（只對增量模式）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="220"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>參考幀變化時運行 FFT；每段內使用熱啟動。是增量模式的典型預設值。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="229"/>
        <source>Previous frame</source>
        <translation>上一幀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="239"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="311"/>
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
        <location filename="../../gui/panels/left_sidebar.py" line="190"/>
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
        <location filename="../../gui/app.py" line="194"/>
        <source>&amp;File</source>
        <translation>檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="196"/>
        <source>Open Session…</source>
        <translation>開啟會話…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>Save Session…</source>
        <translation>儲存會話…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="207"/>
        <source>Quit</source>
        <translation>離開</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="215"/>
        <source>&amp;Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="216"/>
        <source>Language</source>
        <translation>語言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="245"/>
        <source>Language changed</source>
        <translation>語言已切換</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="249"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>語言已切換至 %1。請重啟 pyALDIC 以讓所有界面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="260"/>
        <source>Save Session</source>
        <translation>儲存會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="262"/>
        <location filename="../../gui/app.py" line="289"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC 會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <location filename="../../gui/app.py" line="291"/>
        <source>All Files</source>
        <translation>全部檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="273"/>
        <source>Save Session Failed</source>
        <translation>儲存會話失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="287"/>
        <source>Open Session</source>
        <translation>開啟會話</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>JSON</source>
        <translation>JSON</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Open Session Failed</source>
        <translation>開啟會話失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="651"/>
        <location filename="../../gui/app.py" line="704"/>
        <source>Load images first.</source>
        <translation>請先載入影像。</translation>
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
        <location filename="../../gui/widgets/param_panel.py" line="82"/>
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
        <location filename="../../gui/widgets/param_panel.py" line="88"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>沿感興趣區域的外部邊界局部加密網格。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>加密強度。最小單元尺寸 = max(2, 子集步長 / 2^級別)。對內部邊界、外部邊界和畫筆加密區域統一生效。可用級別取決於子集尺寸和步長。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>加密級別</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="173"/>
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
        <location filename="../../gui/widgets/param_panel.py" line="181"/>
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
        <location filename="../../gui/widgets/param_panel.py" line="241"/>
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
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>採集幀率（用於速度場）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="161"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/幀</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="348"/>
        <source>Building pipeline configuration...</source>
        <translation>正在構建流水線配置…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="500"/>
        <source>Loading images...</source>
        <translation>正在載入影像…</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="64"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>請先載入影像，再在第 1 幀上繪製感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="72"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累積模式&lt;/b&gt; — 只有第 1 幀需要感興趣區域。後續幀都直接與其比較。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="82"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;增量模式，每幀&lt;/b&gt; — 第 1 幀需要感興趣區域。系統會自動將其扭曲到每個後續幀（無需逐幀繪製）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="99"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，每 %1 幀&lt;/b&gt; — 請在以下幀繪製感興趣區域：&lt;b&gt;%2&lt;/b&gt;（共 %3 個參考幀）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="113"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;增量模式，自定義&lt;/b&gt; — 未設定自定義參考幀。僅第 1 幀為參考；請在參考幀清單中添加更多索引。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="123"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，自定義&lt;/b&gt; — 請在以下幀繪製感興趣區域：&lt;b&gt;%1&lt;/b&gt;（共 %2 個參考幀）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="129"/>
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
        <location filename="../../gui/widgets/roi_toolbar.py" line="76"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>向感興趣區域添加形狀（多邊形 / 矩形 / 圓形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="83"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>從感興趣區域裁剪形狀（多邊形 / 矩形 / 圓形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 加密</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="93"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>用畫筆繪製額外的網格加密區域
（僅在第 1 幀可用 — 網格點會自動扭曲到後續幀）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="98"/>
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
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="58"/>
        <source>Run DIC Analysis</source>
        <translation>運行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="71"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="78"/>
        <source>Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).</source>
        <translation>取消當前分析。已計算的幀會保留；運行狀態標記為 IDLE（非 DONE）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="84"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="92"/>
        <source>Open Strain Window</source>
        <translation>開啟應變窗口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="97"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在獨立的後處理窗口中計算並可視化應變。需先完成一次運行以獲得位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="104"/>
        <source>PROGRESS</source>
        <translation>進度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <location filename="../../gui/panels/right_sidebar.py" line="350"/>
        <source>Ready</source>
        <translation>就緒</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="121"/>
        <location filename="../../gui/panels/right_sidebar.py" line="352"/>
        <location filename="../../gui/panels/right_sidebar.py" line="406"/>
        <source>ELAPSED  %1</source>
        <translation>已用  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="126"/>
        <location filename="../../gui/panels/right_sidebar.py" line="354"/>
        <location filename="../../gui/panels/right_sidebar.py" line="414"/>
        <location filename="../../gui/panels/right_sidebar.py" line="418"/>
        <source>REMAINING  %1</source>
        <translation>剩餘  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="134"/>
        <source>FIELD</source>
        <translation>場變量</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="141"/>
        <source>Show on deformed frame</source>
        <translation>在變形幀上顯示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="146"/>
        <source>When checked, overlay results on the deformed (current) frame instead of the reference frame</source>
        <translation>勾選後，將結果疊加在變形（當前）幀上，而非參考幀</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="151"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="156"/>
        <source>Colormap</source>
        <translation>色圖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Opacity</source>
        <translation>透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="184"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>疊加圖透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="190"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="202"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="321"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>運行前請在每個紅色區域放置至少一個種子點（紅色 = 需要種子點）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="389"/>
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
        <location filename="../../gui/widgets/strain_navigator.py" line="222"/>
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
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="69"/>
        <source>Plane fitting</source>
        <translation>平面擬合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="70"/>
        <source>FEM nodal</source>
        <translation>有限元節點</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="72"/>
        <source>Method</source>
        <translation>方法</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="81"/>
        <source>VSG size</source>
        <translation>VSG 尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="97"/>
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
        <location filename="../../gui/widgets/strain_param_panel.py" line="98"/>
        <source>Strain field smoothing</source>
        <translation>應變場平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="103"/>
        <source>Infinitesimal</source>
        <translation>無窮小應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="104"/>
        <source>Eulerian</source>
        <translation>歐拉應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>Green-Lagrangian</source>
        <translation>格林-拉格朗日應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="107"/>
        <source>Strain type</source>
        <translation>應變類型</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="62"/>
        <source>Show on deformed frame</source>
        <translation>在變形幀上顯示</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="74"/>
        <source>Auto</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="96"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="98"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="119"/>
        <source>Strain Post-Processing</source>
        <translation>應變後處理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="165"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="166"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到視口</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="172"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="173"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="176"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="181"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="223"/>
        <source>STRAIN PARAMETERS</source>
        <translation>應變參數</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="239"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>將位移和應變結果匯出為 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="270"/>
        <source>FIELD</source>
        <translation>場變量</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="276"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="283"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="288"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="453"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 參數已變更 — 請點擊“計算應變”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="182"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="229"/>
        <source>Compute Strain</source>
        <translation>計算應變</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="235"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="378"/>
        <source>Starting…</source>
        <translation>啟動中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="400"/>
        <source>Complete</source>
        <translation>完成</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="46"/>
        <source>Use physical units</source>
        <translation>使用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="68"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="83"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="62"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="116"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="141"/>
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
</TS>
