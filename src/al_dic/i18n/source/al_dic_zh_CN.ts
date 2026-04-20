<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN" sourcelanguage="en_US">
<context>
    <name>AdvancedTuningWidget</name>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="39"/>
        <source>ADMM Iterations</source>
        <translation>ADMM 迭代次数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="50"/>
        <source>Number of ADMM alternating minimization cycles for AL-DIC.
1 = single global pass (fastest), 3 = default,
5+ = diminishing returns for most cases.</source>
        <translation>AL-DIC 的 ADMM 交替最小化循环次数。
1 = 单次全局迭代（最快），3 = 默认值，
5 及以上对大多数场景收益递减。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="56"/>
        <source>Only affects AL-DIC solver. Ignored by Local DIC.</source>
        <translation>仅对 AL-DIC 求解器生效，Local DIC 会忽略。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="69"/>
        <source>Auto-expand FFT search on clipped peaks</source>
        <translation>峰值被截断时自动扩大 FFT 搜索范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/advanced_tuning_widget.py" line="75"/>
        <source>When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).

Only relevant for the FFT init-guess mode.</source>
        <translation>当 NCC 峰值触及搜索区域边缘时，自动以更大的搜索范围重试（最大到图像一半尺寸，每次放大 2 倍，共 6 次重试）。

仅对 FFT 初始猜测模式有效。</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="45"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批量导入感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="64"/>
        <source>Mask Folder:</source>
        <translation>掩模文件夹：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="65"/>
        <source>(none)</source>
        <translation>（无）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="70"/>
        <source>Browse...</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="80"/>
        <source>Available Masks</source>
        <translation>可用掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="87"/>
        <source>Auto-Match by Name</source>
        <translation>按文件名自动匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="89"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根据文件名中的数字把掩模文件匹配到对应帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="93"/>
        <source>Assign Sequential</source>
        <translation>顺序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="95"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>从第 0 帧起按顺序把掩模分配给各帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="103"/>
        <source>Frame Assignments</source>
        <translation>帧分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="106"/>
        <source>Mask</source>
        <translation>掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="116"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所选 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="118"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>将所选掩模与所选帧配对</translation>
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
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1108"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1113"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1114"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
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
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1138"/>
        <source>Show Grid</source>
        <translation>显示网格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1139"/>
        <source>Show/hide computational mesh grid</source>
        <translation>显示/隐藏计算网格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1142"/>
        <source>Show Subset</source>
        <translation>显示子集</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1143"/>
        <source>Show subset window on hover (requires Grid)</source>
        <translation>悬停时显示子集窗口（需要先开启网格）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1397"/>
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
        <translation>种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="113"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="115"/>
        <source>FFT every frame</source>
        <translation>每帧 FFT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="116"/>
        <source>FFT every %1 fr</source>
        <translation>每 %1 帧 FFT</translation>
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
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="30"/>
        <source>Auto</source>
        <translation>自动</translation>
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
        <translation>全选</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="471"/>
        <source>None</source>
        <translation>全不选</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="500"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="520"/>
        <source>OUTPUT FOLDER</source>
        <translation>输出文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="528"/>
        <source>Select output folder…</source>
        <translation>选择输出文件夹…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="532"/>
        <source>Browse…</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="537"/>
        <source>Open Folder</source>
        <translation>打开文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="545"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="549"/>
        <source>Enable physical units</source>
        <translation>启用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="554"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸缩放位移值，并在色条标签显示物理单位。应变为无量纲，不受影响。</translation>
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
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="590"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="591"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="592"/>
        <source>Animation</source>
        <translation>动画</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="593"/>
        <source>Report</source>
        <translation>报告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="613"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="615"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 归档 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="617"/>
        <source>MATLAB (.mat)</source>
        <translation>MATLAB (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="619"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐帧）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="622"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ：逐帧一个文件（默认：合并为单个文件）</translation>
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
        <translation>选择：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>STRAIN</source>
        <translation>应变</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="657"/>
        <source>Run Compute Strain first.</source>
        <translation>请先运行“计算应变”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="684"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 参数文件（JSON）始终导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="690"/>
        <source>Export Data</source>
        <translation>导出数据</translation>
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
        <source>FRAME 0/0</source>
        <translation>第 0/0 帧</translation>
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
        <translation>编辑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="241"/>
        <source>Need</source>
        <translation>待绘</translation>
    </message>
</context>
<context>
    <name>InitGuessWidget</name>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="111"/>
        <source>Starting Points</source>
        <translation>种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="122"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="131"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="313"/>
        <source>Place Starting Points</source>
        <translation>放置种子点</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="136"/>
        <source>Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.</source>
        <translation>在画布上进入放置模式。左键添加、右键删除，按 Esc 或再次点击退出。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="138"/>
        <source>Auto-place</source>
        <translation>自动放置</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="142"/>
        <source>Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.</source>
        <translation>在每个空区域填入 NCC 最高的节点。已有种子点会保留。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="144"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="148"/>
        <source>Remove every Starting Point. Faster than right-clicking each one individually.</source>
        <translation>移除所有种子点。比逐个右键删除快。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="153"/>
        <location filename="../../gui/widgets/init_guess_widget.py" line="323"/>
        <source>%1 / %2 regions ready</source>
        <translation>%1 / %2 区域就绪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="172"/>
        <source>FFT (cross-correlation)</source>
        <translation>FFT（互相关）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="183"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="194"/>
        <source>Every</source>
        <translation>每</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="199"/>
        <source>Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N &gt; 1 uses warm-start between resets to limit error propagation to N frames.</source>
        <translation>每 N 帧运行一次 FFT。N = 1 表示每帧都做 FFT（最安全但最慢）。N &gt; 1 在两次重置之间使用热启动，将误差传播限制在 N 帧内。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="206"/>
        <source>(N=1 = every frame)</source>
        <translation>（N=1 即每帧）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="216"/>
        <source>Only when reference frame updates (incremental only)</source>
        <translation>仅在参考帧更新时（只对增量模式）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="220"/>
        <source>Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.</source>
        <translation>参考帧变化时运行 FFT；每段内使用热启动。是增量模式的典型默认值。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="229"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/init_guess_widget.py" line="239"/>
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
        <location filename="../../gui/widgets/init_guess_widget.py" line="311"/>
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
        <location filename="../../gui/panels/left_sidebar.py" line="190"/>
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
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="194"/>
        <source>&amp;File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="196"/>
        <source>Open Session…</source>
        <translation>打开会话…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>Save Session…</source>
        <translation>保存会话…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="207"/>
        <source>Quit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="215"/>
        <source>&amp;Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="216"/>
        <source>Language</source>
        <translation>语言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="245"/>
        <source>Language changed</source>
        <translation>语言已切换</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="249"/>
        <source>Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.</source>
        <translation>语言已切换至 %1。请重启 pyALDIC 以让所有界面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="260"/>
        <source>Save Session</source>
        <translation>保存会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="262"/>
        <location filename="../../gui/app.py" line="289"/>
        <source>pyALDIC Session</source>
        <translation>pyALDIC 会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="263"/>
        <location filename="../../gui/app.py" line="291"/>
        <source>All Files</source>
        <translation>全部文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="273"/>
        <source>Save Session Failed</source>
        <translation>保存会话失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="287"/>
        <source>Open Session</source>
        <translation>打开会话</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>JSON</source>
        <translation>JSON</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>Open Session Failed</source>
        <translation>打开会话失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="651"/>
        <location filename="../../gui/app.py" line="704"/>
        <source>Load images first.</source>
        <translation>请先加载图像。</translation>
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
        <location filename="../../gui/widgets/param_panel.py" line="37"/>
        <source>Subset Size</source>
        <translation>子集尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="43"/>
        <source>IC-GN subset window size in pixels (odd number)</source>
        <translation>IC-GN 子集窗口尺寸（像素，奇数）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="50"/>
        <source>Subset Step</source>
        <translation>子集步长</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="53"/>
        <source>Node spacing in pixels (must be power of 2)</source>
        <translation>节点间距（像素，必须是 2 的幂）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="61"/>
        <location filename="../../gui/widgets/param_panel.py" line="186"/>
        <source>Search Range</source>
        <translation>搜索范围</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Refine Inner Boundary</source>
        <translation>加密内部边界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="82"/>
        <source>Locally refine the mesh along internal mask boundaries
(holes inside the Region of Interest). Useful for bubble / void edges.</source>
        <translation>沿内部掩模边界局部加密网格
（感兴趣区域内部的孔洞）。适合气泡 / 空洞边缘。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="83"/>
        <source>Refine Outer Boundary</source>
        <translation>加密外部边界</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="88"/>
        <source>Locally refine the mesh along the outer Region of Interest
boundary.</source>
        <translation>沿感兴趣区域的外部边界局部加密网格。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.</source>
        <translation>加密强度。最小单元尺寸 = max(2, 子集步长 / 2^级别)。对内部边界、外部边界和画笔加密区域统一生效。可用级别取决于子集尺寸和步长。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="108"/>
        <source>Refinement Level</source>
        <translation>加密级别</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="173"/>
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
        <location filename="../../gui/widgets/param_panel.py" line="181"/>
        <source>Initial half-width (pixels) of the single-point NCC search at each Starting Point.
Auto-expands 2x per retry if the peak is clipped, up to image half-size.
Only affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).</source>
        <translation>每个种子点处单点 NCC 搜索的初始半宽（像素）。
若峰值被截断，每次重试自动放大 2 倍，最大到图像一半尺寸。
仅影响种子点引导；其他节点使用 F-aware 传播（无需逐节点搜索）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Initial Seed Search</source>
        <translation>初始种子搜索</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="241"/>
        <source>min element size = %1 px  (subset_step=%2, level=%3)</source>
        <translation>最小单元尺寸 = %1 px  (子集步长=%2, 级别=%3)</translation>
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
        <location filename="../../gui/widgets/physical_units_widget.py" line="93"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>采集帧率（用于速度场）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="161"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/帧</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="348"/>
        <source>Building pipeline configuration...</source>
        <translation>正在构建流水线配置…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="500"/>
        <source>Loading images...</source>
        <translation>正在加载图像…</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="64"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>请先加载图像，再在第 1 帧上绘制感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="72"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累积模式&lt;/b&gt; — 只有第 1 帧需要感兴趣区域。后续帧都直接与其比较。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="82"/>
        <source>&lt;b&gt;Incremental, every frame&lt;/b&gt; — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).</source>
        <translation>&lt;b&gt;增量模式，每帧&lt;/b&gt; — 第 1 帧需要感兴趣区域。系统会自动将其扭曲到每个后续帧（无需逐帧绘制）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="99"/>
        <source>&lt;b&gt;Incremental, every %1 frames&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%2&lt;/b&gt; (%3 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，每 %1 帧&lt;/b&gt; — 请在以下帧绘制感兴趣区域：&lt;b&gt;%2&lt;/b&gt;（共 %3 个参考帧）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="113"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.</source>
        <translation>&lt;b&gt;增量模式，自定义&lt;/b&gt; — 未设置自定义参考帧。仅第 1 帧为参考；请在参考帧列表中添加更多索引。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="123"/>
        <source>&lt;b&gt;Incremental, custom&lt;/b&gt; — draw a Region of Interest on frames: &lt;b&gt;%1&lt;/b&gt; (%2 reference frames total).</source>
        <translation>&lt;b&gt;增量模式，自定义&lt;/b&gt; — 请在以下帧绘制感兴趣区域：&lt;b&gt;%1&lt;/b&gt;（共 %2 个参考帧）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="129"/>
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
        <location filename="../../gui/widgets/roi_toolbar.py" line="76"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>向感兴趣区域添加形状（多边形 / 矩形 / 圆形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="83"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>从感兴趣区域裁剪形状（多边形 / 矩形 / 圆形）</translation>
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
        <translation>用画笔绘制额外的网格加密区域
（仅在第 1 帧可用 — 网格点会自动扭曲到后续帧）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="98"/>
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
        <translation>反转感兴趣区域掩模</translation>
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
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="58"/>
        <source>Run DIC Analysis</source>
        <translation>运行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="71"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="78"/>
        <source>Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).</source>
        <translation>取消当前分析。已计算的帧会保留；运行状态标记为 IDLE（非 DONE）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="84"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="92"/>
        <source>Open Strain Window</source>
        <translation>打开应变窗口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="97"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在独立的后处理窗口中计算并可视化应变。需先完成一次运行以获得位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="104"/>
        <source>PROGRESS</source>
        <translation>进度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <location filename="../../gui/panels/right_sidebar.py" line="350"/>
        <source>Ready</source>
        <translation>就绪</translation>
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
        <translation>剩余  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="134"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="141"/>
        <source>Show on deformed frame</source>
        <translation>在变形帧上显示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="146"/>
        <source>When checked, overlay results on the deformed (current) frame instead of the reference frame</source>
        <translation>勾选后，将结果叠加在变形（当前）帧上，而非参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="151"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="156"/>
        <source>Colormap</source>
        <translation>色图</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Opacity</source>
        <translation>透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="184"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>叠加图透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="190"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="202"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="321"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>运行前请在每个红色区域放置至少一个种子点（红色 = 需要种子点）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="389"/>
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
        <location filename="../../gui/widgets/strain_navigator.py" line="222"/>
        <source>FRAME 0/0</source>
        <translation>第 0/0 帧</translation>
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
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="69"/>
        <source>Plane fitting</source>
        <translation>平面拟合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="70"/>
        <source>FEM nodal</source>
        <translation>有限元节点</translation>
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
        <translation>计算后对应变场做高斯平滑。
σ 为高斯核宽度；“step” 为 DIC 节点间距。
  Light（0.5 × step）：轻度平滑，保留细节。
  Medium（1 × step）：平衡选择，推荐用于噪声数据。
  Strong（2 × step）⚠：强平滑，可能模糊真实梯度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="98"/>
        <source>Strain field smoothing</source>
        <translation>应变场平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="103"/>
        <source>Infinitesimal</source>
        <translation>无穷小应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="104"/>
        <source>Eulerian</source>
        <translation>欧拉应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>Green-Lagrangian</source>
        <translation>格林-拉格朗日应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="107"/>
        <source>Strain type</source>
        <translation>应变类型</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="62"/>
        <source>Show on deformed frame</source>
        <translation>在变形帧上显示</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="74"/>
        <source>Auto</source>
        <translation>自动</translation>
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
        <translation>应变后处理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="165"/>
        <source>Fit</source>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="166"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="172"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="173"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
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
        <translation>应变参数</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="239"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>将位移和应变结果导出为 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="270"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="276"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="283"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="288"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="453"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 参数已变更 — 请点击“计算应变”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="182"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="229"/>
        <source>Compute Strain</source>
        <translation>计算应变</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="235"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="378"/>
        <source>Starting…</source>
        <translation>启动中…</translation>
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
        <translation>使用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="68"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="83"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="62"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="116"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>增量追踪中参考帧的刷新策略。
每帧：每帧都更新参考（单步位移最小，
对大变形最稳健）。
每 N 帧：每 N 帧更新一次（速度与稳健性的折中）。
自定义：由用户指定参考帧索引列表。</translation>
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
        <location filename="../../gui/widgets/workflow_type_panel.py" line="141"/>
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
</TS>
