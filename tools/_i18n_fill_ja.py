"""Fill al_dic_ja.ts with Japanese translations.

Terminology conventions (see docs/i18n/glossary.md):
  * DIC technical terms: サブセット / 変位 / ひずみ / メッシュ /
    シード点 / 参照フレーム / 適応的細分化
  * Proper nouns kept English: pyALDIC, AL-DIC, IC-GN, ADMM, FFT,
    NCC, FEM, Q8, ROI, POI, RMSE, NumPy, MATLAB, CSV, NPZ, PNG, PDF
  * Placeholders (%1, %2, %n, HTML tags <b>) preserved verbatim
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_ja.ts"

TRANSLATIONS: dict[str, str] = {
    # MainWindow chrome
    "&Settings": "設定",
    "Language": "言語",
    "Language changed": "言語を変更しました",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        "言語を %1 に変更しました。すべての画面に反映するには pyALDIC を再起動してください。",
    "&File": "ファイル",
    "Open Session…": "セッションを開く…",
    "Save Session…": "セッションを保存…",
    "Quit": "終了",
    "Save Session": "セッションを保存",
    "Open Session": "セッションを開く",
    "Save Session Failed": "セッションの保存に失敗しました",
    "Open Session Failed": "セッションを開けませんでした",
    "pyALDIC Session": "pyALDIC セッション",
    "All Files": "すべてのファイル",
    "JSON": "JSON",

    # Right sidebar — run controls
    "Run DIC Analysis": "DIC 解析を実行",
    "Cancel": "キャンセル",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "現在の解析をキャンセルします。計算済みのフレームは保持され、ジョブは IDLE（DONE ではない）となります。",
    "Export Results": "結果をエクスポート",
    "Open Strain Window": "ひずみウィンドウを開く",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "別ウィンドウでひずみを計算・可視化します。完了した実行結果の変位データが必要です。",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "実行前に、各赤色領域に少なくとも 1 つのシード点を配置してください(赤色 = シード点が必要)。",

    # Progress / Field / Visualization
    "PROGRESS": "進捗",
    "Ready": "準備完了",
    "ELAPSED  %1": "経過  %1",
    "REMAINING  %1": "残り  %1",
    "%1  —  Frame %2": "%1  —  フレーム %2",
    "FIELD": "表示項目",
    "Show on deformed frame": "変形後フレームに表示",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "オンにすると、結果を参照フレームではなく変形後(現在)フレームに重ねて表示します",
    "VISUALIZATION": "可視化",
    "Colormap": "カラーマップ",
    "Opacity": "不透明度",
    "Overlay opacity (0 = transparent, 100 = opaque)":
        "オーバーレイの不透明度(0 = 透明、100 = 不透明)",
    "PHYSICAL UNITS": "物理単位",
    "LOG": "ログ",
    "Clear": "クリア",

    # Left sidebar
    "IMAGES": "画像",
    "Drop image folder\nor Browse": "画像フォルダをドロップ\nまたは参照",
    "Select Image Folder": "画像フォルダを選択",
    "Natural Sort (1, 2, …, 10)": "自然順ソート (1, 2, …, 10)",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "ファイル名中の数字順でソート: image1, image2, …, image10\nデフォルト(オフ): 辞書順 — ゼロ埋めされた名前に適しています",
    "WORKFLOW TYPE": "ワークフロー種別",
    "INITIAL GUESS": "初期推定",
    "REGION OF INTEREST": "関心領域",
    "PARAMETERS": "パラメータ",
    "ADVANCED": "詳細設定",

    # Workflow type panel
    "Tracking Mode": "追跡モード",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "逐次: 各フレームを直前の参照フレームと比較します。\n大きな累積変形に適し、大回転では必須です。\n\n累積: 各フレームを第 1 フレームと比較します。\n小さく単調な変形にのみ適します。",
    "Solver": "ソルバー",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC: 独立サブセットマッチング(IC-GN)。高速で\n局所特徴を保持します。小変形や高品質画像に最適です。\n\nAL-DIC: 全体 FEM 正則化付き拡張ラグランジュ。\nサブセット間の変位適合性を強制します。\n大変形・ノイズ画像・ひずみ精度重視の場合に最適です。",
    "Reference Update": "参照フレーム更新",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "逐次追跡中の参照フレームの更新タイミング。\n毎フレーム: 毎フレーム参照をリセット(ステップ変位最小、\n大変形に最もロバスト)。\nN フレームごと: N フレームごとにリセット(速度と頑健性のバランス)。\nカスタム: ユーザ指定の参照フレームインデックス。",
    "Update reference every N frames": "N フレームごとに参照を更新",
    "Interval": "間隔",
    "Reference Frames": "参照フレーム",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "参照フレームとして使用するフレーム番号(0 始まり、カンマ区切り)",

    # ROI toolbar
    "+ Add": "+ 追加",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "関心領域に形状を追加します(多角形 / 矩形 / 円)",
    "Cut": "切り取り",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "関心領域から形状を切り取ります(多角形 / 矩形 / 円)",
    "+ Refine": "+ 細分化",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "ブラシで追加の細分化領域を塗ります\n(フレーム 1 のみ — 後続フレームへ自動ワープされます)",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "細分化ブラシはフレーム 1 でのみ使用可能です。フレーム 1 に切り替えて領域を塗ってください。後続フレームには自動でワープされます。",
    "Import": "インポート",
    "Import mask from image file": "画像ファイルからマスクをインポート",
    "Batch Import": "一括インポート",
    "Batch import mask files for multiple frames": "複数フレームのマスクファイルを一括インポート",
    "Save": "保存",
    "Save current mask to PNG file": "現在のマスクを PNG ファイルに保存",
    "Invert": "反転",
    "Invert the Region of Interest mask": "関心領域マスクを反転",
    "Clear all Region of Interest masks": "すべての関心領域マスクをクリア",
    "Radius": "半径",
    "Paint": "塗り",
    "Erase": "消去",
    "Clear Brush": "ブラシをクリア",

    # Parameters panel
    "Subset Size": "サブセットサイズ",
    "IC-GN subset window size in pixels (odd number)":
        "IC-GN サブセットウィンドウサイズ(ピクセル、奇数)",
    "Subset Step": "サブセットステップ",
    "Node spacing in pixels (must be power of 2)":
        "ノード間隔(ピクセル、2 の累乗)",
    "Search Range": "探索範囲",
    "Initial Seed Search": "初期シード探索",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "FFT 探索で検出可能な 1 フレームあたりの最大変位(ピクセル)。\n想定されるフレーム間動きより十分大きく設定してください。\n逐次モードで大回転が発生する場合、次を満たす必要があります:\n  半径 × sin(1 ステップ角)。",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "各シード点における単点 NCC 探索の初期半径(ピクセル)。\nピークが打ち切られた場合、画像半サイズまで 2 倍ずつ自動拡大します。\nシード点の初期化にのみ影響し、他のノードは F-aware 伝播(ノード単位の探索なし)を使用します。",
    "Refine Inner Boundary": "内部境界を細分化",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "内部マスク境界に沿ってメッシュを局所的に細分化します\n(関心領域内の穴)。気泡や空隙の縁に有用です。",
    "Refine Outer Boundary": "外部境界を細分化",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "関心領域の外部境界に沿ってメッシュを局所的に細分化します。",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "細分化の強さ。最小要素サイズ = max(2, サブセットステップ / 2^レベル)。内部・外部境界およびブラシで塗った領域すべてに一律適用されます。利用可能なレベルはサブセットサイズとステップに依存します。",
    "Refinement Level": "細分化レベル",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "最小要素サイズ = %1 px  (サブセットステップ=%2, レベル=%3)",

    # Initial guess widget
    "Starting Points": "シード点",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "シード点をいくつか配置します。pyALDIC は単点 NCC で初期化し、メッシュ隣接に沿って場を伝播します。\n\n最適な場面:\n• 大きなフレーム間変位(> 50 px)\n• 不連続な場(亀裂、せん断帯)\n• FFT が誤ピークを選ぶケース\n\nROI 作成/編集時に領域ごとに自動配置されます。",
    "Place Starting Points": "シード点を配置",
    "Placing... (click to exit)": "配置中…(クリックで終了)",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "キャンバスで配置モードに入ります。左クリックで追加、右クリックで削除、Esc または再クリックで終了。",
    "Auto-place": "自動配置",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "各空領域に最高 NCC のノードを配置します。既存のシード点は保持されます。",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "すべてのシード点を削除します。1 つずつ右クリックするより高速です。",
    "%1 / %2 regions ready": "%1 / %2 領域 準備完了",
    "FFT (cross-correlation)": "FFT(相互相関)",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "格子全体の正規化相互相関。探索半径内で頑健、ピークが打ち切られたら自動拡大します。\n\n最適な場面:\n• 小〜中程度の滑らかな動き\n• 良好なスペックル\n• 特別な設定が不要\n\nコストが探索半径とともに増加するため、非常に大きな変位では遅くなります。",
    "Every": "毎",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "N フレームごとに FFT を実行します。N = 1 で毎フレーム FFT(最も安全・低速)。N > 1 ではリセット間でウォームスタートし、誤差伝播を N フレーム以内に抑えます。",
    "(N=1 = every frame)": "(N=1 は毎フレーム)",
    "Only when reference frame updates (incremental only)":
        "参照フレーム更新時のみ(逐次のみ)",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "参照フレームが変わるたびに FFT を実行し、区間内ではウォームスタートします。逐次モードの標準設定です。",
    "Previous frame": "前フレーム",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "前フレームの収束変位を初期推定として使用します。相互相関は実行しません。\n\n最適な場面:\n• 非常に小さなフレーム間動き(数ピクセル)\n• 動きが滑らかな場合の最速オプション\n\n長いシーケンスでは誤差が累積します。ノイズの多いデータや動きが大きい場合は FFT またはシード点を推奨します。",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "まず画像を読み込み、その後フレーム 1 に関心領域を描画してください。",
    "<b>Accumulative mode</b> — only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>累積モード</b> — 関心領域はフレーム 1 にのみ必要です。後続フレームはすべて直接比較されます。",
    "<b>Incremental, every frame</b> — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>逐次、毎フレーム</b> — フレーム 1 に関心領域が必要です。後続フレームには自動で前方ワープされます(フレーム単位の描画は不要)。",
    "<b>Incremental, every %1 frames</b> — draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>逐次、%1 フレームごと</b> — 次のフレームに関心領域を描画してください: <b>%2</b>(参照フレーム合計 %3)。",
    "<b>Incremental, custom</b> — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>逐次、カスタム</b> — カスタム参照フレーム未設定。フレーム 1 が唯一の参照となります。参照フレーム欄にインデックスを追加してください。",
    "<b>Incremental, custom</b> — draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>逐次、カスタム</b> — 次のフレームに関心領域を描画してください: <b>%1</b>(参照フレーム合計 %2)。",
    "Draw a Region of Interest on frame 1.": "フレーム 1 に関心領域を描画してください。",

    # Export dialog
    "All": "全選択",
    "None": "全解除",
    "OUTPUT FOLDER": "出力フォルダ",
    "Select output folder…": "出力フォルダを選択…",
    "Browse…": "参照…",
    "Open Folder": "フォルダを開く",
    "Enable physical units": "物理単位を有効化",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "変位値をピクセルサイズでスケールし、カラーバーのラベルに物理単位を表示します。ひずみは無次元量のため影響を受けません。",
    "/ pixel": "/ ピクセル",
    "Pixel size": "ピクセルサイズ",
    "fps": "fps",
    "Frame rate": "フレームレート",
    "Data": "データ",
    "Images": "画像",
    "Animation": "アニメーション",
    "Report": "レポート",
    "FORMAT": "形式",
    "NumPy Archive (.npz)": "NumPy アーカイブ (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV(フレーム単位)",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ: フレーム単位で 1 ファイル(デフォルト: 統合 1 ファイル)",
    "DISPLACEMENT": "変位",
    "Select:": "選択:",
    "STRAIN": "ひずみ",
    "Run Compute Strain first.": "先に「ひずみを計算」を実行してください。",
    "✓ Parameters file (JSON) always exported": "✓ パラメータファイル (JSON) は常にエクスポートされます",
    "Export Data": "データをエクスポート",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "関心領域マスクを一括インポート",
    "Mask Folder:": "マスクフォルダ:",
    "(none)": "(なし)",
    "Browse...": "参照…",
    "Available Masks": "利用可能なマスク",
    "Auto-Match by Name": "ファイル名で自動マッチ",
    "Match mask files to frames by number in filename":
        "ファイル名中の番号でマスクをフレームに対応付けます",
    "Assign Sequential": "順次割り当て",
    "Assign masks to frames in order starting from frame 0":
        "フレーム 0 から順番にマスクを割り当てます",
    "Frame Assignments": "フレーム割り当て",
    "Frame": "フレーム",
    "Image": "画像",
    "Mask": "マスク",
    "Assign Selected ->": "選択項目を割り当て ->",
    "Pair selected mask(s) with selected frame(s)":
        "選択したマスクと選択したフレームを対応付けます",
    "Clear All": "すべてクリア",

    # Canvas area / toolbar
    "Fit": "フィット",
    "Fit image to viewport": "画像をビューポートに合わせる",
    "100%": "100%",
    "Zoom to 100% (1:1)": "100% (1:1) ズーム",
    "Zoom in": "拡大",
    "–": "–",
    "Zoom out": "縮小",
    "Show Grid": "グリッドを表示",
    "Show/hide computational mesh grid": "計算メッシュグリッドの表示/非表示",
    "Show Subset": "サブセットを表示",
    "Show subset window on hover (requires Grid)":
        "ホバー時にサブセットウィンドウを表示(グリッド必須)",
    "Placing Starting Points": "シード点を配置中",

    # Color range
    "Range": "範囲",
    "Auto": "自動",
    "Min": "最小",
    "Max": "最大",

    # Field selector
    "Disp U": "変位 U",
    "Disp V": "変位 V",

    # Frame navigator / strain navigator
    "Play animation": "アニメーションを再生",
    "▶": "▶",
    "Next frame": "次のフレーム",
    "Playback speed": "再生速度",
    "FRAME 0/0": "フレーム 0/0",
    "⏸": "⏸",
    "Pause animation": "アニメーションを一時停止",

    # Image list
    "Add": "追加",
    "Edit": "編集",
    "Need": "未設定",

    # Mesh appearance
    "Mesh color": "メッシュ色",
    "Click to choose mesh line color": "クリックしてメッシュ線の色を選択",
    "Line width": "線幅",

    # Physical units widget
    "Use physical units": "物理単位を使用",
    "Physical size of one image pixel": "1 画像ピクセルの物理サイズ",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)":
        "取得フレームレート(速度場に使用)",
    "Disp: px  Velocity: px/fr": "変位: px  速度: px/fr",

    # Advanced tuning
    "ADMM Iterations": "ADMM 反復回数",
    "Number of ADMM alternating minimization cycles for AL-DIC.\n1 = single global pass (fastest), 3 = default,\n5+ = diminishing returns for most cases.":
        "AL-DIC の ADMM 交互最小化の反復回数。\n1 = 単一グローバルパス(最速)、3 = デフォルト、\n5 以上はほとんどのケースで収穫逓減。",
    "Only affects AL-DIC solver. Ignored by Local DIC.":
        "AL-DIC ソルバーにのみ影響します。Local DIC では無視されます。",
    "Auto-expand FFT search on clipped peaks":
        "ピークが打ち切られたら FFT 探索を自動拡大",
    "When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).\n\nOnly relevant for the FFT init-guess mode.":
        "NCC ピークが探索領域の端に達したら、より広い領域で自動的に再試行します(最大で画像半分まで、2 倍ずつ 6 回)。\n\nFFT 初期推定モードでのみ有効です。",

    # Canvas overlay
    "Mode": "モード",
    "Init": "初期推定",
    "Accumulative": "累積式",
    "Incremental": "逐次式",
    "Local DIC": "Local DIC",
    "Every Frame": "毎フレーム",
    "Every N Frames": "N フレームごと",
    "Custom Frames": "カスタムフレーム",
    "ADMM (%1 iter)": "ADMM (%1 反復)",
    "FFT every frame": "毎フレーム FFT",
    "FFT every %1 fr": "%1 フレームごと FFT",
    "FFT": "FFT",

    # Log messages
    "Load images first.": "先に画像を読み込んでください。",
    "Building pipeline configuration...": "パイプライン設定を構築中…",
    "Loading images...": "画像を読み込み中…",

    # Strain window
    "Strain Post-Processing": "ひずみ後処理",
    "Compute Strain": "ひずみを計算",
    "Starting…": "開始中…",
    "Complete": "完了",
    "⚠ Params changed -- click Compute Strain": "⚠ パラメータが変更されました — 「ひずみを計算」をクリックしてください",
    "Unit: px/frame": "単位: px/frame",

    # Strain field selector
    "STRAIN PARAMETERS": "ひずみパラメータ",

    # Strain param panel
    "Method": "手法",
    "Plane fitting": "平面フィッティング",
    "FEM nodal": "FEM 節点",
    "VSG size": "VSG サイズ",
    "Strain field smoothing": "ひずみ場の平滑化",
    "Strain type": "ひずみ種別",
    "Infinitesimal": "微小ひずみ",
    "Eulerian": "オイラーひずみ",
    "Green-Lagrangian": "グリーン-ラグランジュひずみ",
    "Gaussian smoothing of the strain field after computation.\nσ is the Gaussian kernel width; 'step' = DIC node spacing.\n  Light  (0.5 × step):  subtle, preserves fine features.\n  Medium (1 × step):    balanced, recommended for noisy data.\n  Strong (2 × step) ⚠:  aggressive, may blur real gradients.":
        "計算後のひずみ場にガウス平滑化を適用。\nσ はガウスカーネル幅、'step' は DIC ノード間隔。\n  Light  (0.5 × step): 穏やか、細部を保持。\n  Medium (1 × step):    バランス型、ノイズデータに推奨。\n  Strong (2 × step) ⚠: 強め、実勾配をぼかす可能性あり。",

    # Strain window export tooltip
    "Export displacement and strain results to NPZ / MAT / CSV / PNG":
        "変位とひずみ結果を NPZ / MAT / CSV / PNG にエクスポート",
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
    print(f"[ja] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
