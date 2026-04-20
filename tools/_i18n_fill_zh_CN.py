"""One-shot translator: fill zh_CN .ts from a hard-coded mapping.

Not committed as a reusable tool — this is a one-off migration
helper run during the Phase 1 widget retrofit. Future translation
updates should go through Claude Code reading the glossary at
docs/i18n/glossary.md and writing to the .ts directly.
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_zh_CN.ts"

# Mapping: source string -> simplified Chinese translation.
# Terminology follows docs/i18n/glossary.md.
TRANSLATIONS: dict[str, str] = {
    # MainWindow (already translated in previous pass — re-assert)
    "&Settings": "设置",
    "Language": "语言",
    "Language changed": "语言已切换",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        "语言已切换至 %1。请重启 pyALDIC 以让所有界面生效。",
    "&File": "文件",
    "Open Session…": "打开会话…",
    "Save Session…": "保存会话…",
    "Quit": "退出",
    "Save Session": "保存会话",
    "Open Session": "打开会话",
    "Save Session Failed": "保存会话失败",
    "Open Session Failed": "打开会话失败",
    "pyALDIC Session": "pyALDIC 会话",
    "All Files": "全部文件",
    "JSON": "JSON",

    # Right sidebar — run controls
    "Run DIC Analysis": "运行 DIC 分析",
    "Cancel": "取消",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "取消当前分析。已计算的帧会保留；运行状态标记为 IDLE（非 DONE）。",
    "Export Results": "导出结果",
    "Open Strain Window": "打开应变窗口",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "在独立的后处理窗口中计算并可视化应变。需先完成一次运行以获得位移结果。",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "运行前请在每个红色区域放置至少一个种子点（红色 = 需要种子点）。",

    # Progress / Field / Visualization
    "PROGRESS": "进度",
    "Ready": "就绪",
    "ELAPSED  %1": "已用  %1",
    "REMAINING  %1": "剩余  %1",
    "%1  \u2014  Frame %2": "%1  \u2014  第 %2 帧",
    "FIELD": "场",
    "Show on deformed frame": "在变形帧上显示",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "勾选后，将结果叠加在变形（当前）帧上，而非参考帧",
    "VISUALIZATION": "可视化",
    "Colormap": "色图",
    "Opacity": "透明度",
    "Overlay opacity (0 = transparent, 100 = opaque)": "叠加图透明度（0 = 透明，100 = 不透明）",
    "PHYSICAL UNITS": "物理单位",
    "LOG": "日志",
    "Clear": "清除",

    # Left sidebar
    "IMAGES": "图像",
    "Drop image folder\nor Browse": "拖入图像文件夹\n或点击浏览",
    "Select Image Folder": "选择图像文件夹",
    "Natural Sort (1, 2, …, 10)": "自然排序（1, 2, …, 10）",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "按文件名中的数字排序：image1, image2, …, image10\n默认（不勾选）：字典序 — 适合已补零的文件名",
    "WORKFLOW TYPE": "工作流类型",
    "INITIAL GUESS": "初始猜测",
    "REGION OF INTEREST": "感兴趣区域",
    "PARAMETERS": "参数",
    "ADVANCED": "高级",

    # Workflow type panel
    "Tracking Mode": "追踪模式",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "增量式：每帧与前一个参考帧比较。\n适用于大量累积变形，大旋转场景必须使用。\n\n累积式：每帧都与第 1 帧比较。\n仅适用于小的、单调的变形。",
    "Solver": "求解器",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC：独立子集匹配（IC-GN）。速度快，\n保留局部锐利特征。适合小变形\n或高质量图像。\n\nAL-DIC：全局 FEM 正则化的增广拉格朗日方法。\n强制子集间的位移相容性。适合大变形、\n噪声图像，或对应变精度要求高的场景。",
    "Reference Update": "参考帧更新",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "增量追踪中参考帧的刷新策略。\n每帧：每帧都更新参考（单步位移最小，\n对大变形最稳健）。\n每 N 帧：每 N 帧更新一次（速度与稳健性的折中）。\n自定义：由用户指定参考帧索引列表。",
    "Update reference every N frames": "每 N 帧更新一次参考帧",
    "Interval": "间隔",
    "Reference Frames": "参考帧列表",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "用作参考帧的帧索引列表（0 为起始），用逗号分隔",

    # ROI toolbar
    "+ Add": "+ 添加",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "向感兴趣区域添加形状（多边形 / 矩形 / 圆形）",
    "Cut": "裁剪",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "从感兴趣区域裁剪形状（多边形 / 矩形 / 圆形）",
    "+ Refine": "+ 加密",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "用画笔绘制额外的网格加密区域\n（仅在第 1 帧可用 — 网格点会自动扭曲到后续帧）",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "加密画笔仅在第 1 帧可用。切换到第 1 帧后可绘制加密区域；系统会自动将其扭曲到后续帧。",
    "Import": "导入",
    "Import mask from image file": "从图像文件导入掩模",
    "Batch Import": "批量导入",
    "Batch import mask files for multiple frames": "批量导入多帧的掩模文件",
    "Save": "保存",
    "Save current mask to PNG file": "将当前掩模保存为 PNG 文件",
    "Invert": "反选",
    "Invert the Region of Interest mask": "反转感兴趣区域掩模",
    "Clear all Region of Interest masks": "清除所有感兴趣区域掩模",
    "Radius": "半径",
    "Paint": "绘制",
    "Erase": "擦除",
    "Clear Brush": "清除画笔",

    # Parameters panel
    "Subset Size": "子集尺寸",
    "IC-GN subset window size in pixels (odd number)":
        "IC-GN 子集窗口尺寸（像素，奇数）",
    "Subset Step": "子集步长",
    "Node spacing in pixels (must be power of 2)":
        "节点间距（像素，必须是 2 的幂）",
    "Search Range": "搜索范围",
    "Initial Seed Search": "初始种子搜索",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "FFT 搜索可检测到的每帧最大位移（像素）。\n设置值应略大于预期的帧间运动。\n对于增量模式下的大旋转，该值必须覆盖\n  半径 × sin(单步角度)。",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "每个种子点处单点 NCC 搜索的初始半宽（像素）。\n若峰值被截断，每次重试自动放大 2 倍，最大到图像一半尺寸。\n仅影响种子点引导；其他节点使用 F-aware 传播（无需逐节点搜索）。",
    "Refine Inner Boundary": "加密内部边界",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "沿内部掩模边界局部加密网格\n（感兴趣区域内部的孔洞）。适合气泡 / 空洞边缘。",
    "Refine Outer Boundary": "加密外部边界",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "沿感兴趣区域的外部边界局部加密网格。",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "加密强度。最小单元尺寸 = max(2, 子集步长 / 2^级别)。对内部边界、外部边界和画笔加密区域统一生效。可用级别取决于子集尺寸和步长。",
    "Refinement Level": "加密级别",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "最小单元尺寸 = %1 px  (子集步长=%2, 级别=%3)",

    # Initial guess widget
    "Starting Points": "种子点",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "放置若干种子点；pyALDIC 在每个点上运行单点 NCC 引导，然后沿网格邻居传播位移场。\n\n最适合：\n• 大帧间位移（> 50 px）\n• 不连续场（裂纹、剪切带）\n• FFT 容易选错峰的场景\n\n绘制或编辑 ROI 时会为每个区域自动放置。",
    "Place Starting Points": "放置种子点",
    "Placing... (click to exit)": "放置中…（再次点击退出）",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "在画布上进入放置模式。左键添加、右键删除，按 Esc 或再次点击退出。",
    "Auto-place": "自动放置",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "在每个空区域填入 NCC 最高的节点。已有种子点会保留。",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "移除所有种子点。比逐个右键删除快。",
    "%1 / %2 regions ready": "%1 / %2 区域就绪",
    "FFT (cross-correlation)": "FFT（互相关）",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "全网格归一化互相关。在搜索半径内稳健；峰值被截断时搜索自动扩展。\n\n最适合：\n• 小到中等的平滑运动\n• 纹理良好的散斑\n• 不需要用户额外设置\n\n计算成本随搜索半径增长，极大位移会变慢。",
    "Every": "每",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "每 N 帧运行一次 FFT。N = 1 表示每帧都做 FFT（最安全但最慢）。N > 1 在两次重置之间使用热启动，将误差传播限制在 N 帧内。",
    "(N=1 = every frame)": "（N=1 即每帧）",
    "Only when reference frame updates (incremental only)":
        "仅在参考帧更新时（只对增量模式）",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "参考帧变化时运行 FFT；每段内使用热启动。是增量模式的典型默认值。",
    "Previous frame": "前一帧",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "使用前一帧收敛的位移作为初始猜测。不运行任何互相关。\n\n最适合：\n• 非常小的帧间运动（几像素）\n• 运动平滑时速度最快\n\n长序列中误差会累积。数据有噪声或运动较大时请选 FFT 或种子点。",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "请先加载图像，再在第 1 帧上绘制感兴趣区域。",
    "<b>Accumulative mode</b> \u2014 only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>累积模式</b> \u2014 只有第 1 帧需要感兴趣区域。后续帧都直接与其比较。",
    "<b>Incremental, every frame</b> \u2014 frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>增量模式，每帧</b> \u2014 第 1 帧需要感兴趣区域。系统会自动将其扭曲到每个后续帧（无需逐帧绘制）。",
    "<b>Incremental, every %1 frames</b> \u2014 draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>增量模式，每 %1 帧</b> \u2014 请在以下帧绘制感兴趣区域：<b>%2</b>（共 %3 个参考帧）。",
    "<b>Incremental, custom</b> \u2014 no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>增量模式，自定义</b> \u2014 未设置自定义参考帧。仅第 1 帧为参考；请在参考帧列表中添加更多索引。",
    "<b>Incremental, custom</b> \u2014 draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>增量模式，自定义</b> \u2014 请在以下帧绘制感兴趣区域：<b>%1</b>（共 %2 个参考帧）。",
    "Draw a Region of Interest on frame 1.": "请在第 1 帧绘制感兴趣区域。",

    # Export dialog
    "All": "全选",
    "None": "全不选",
    "OUTPUT FOLDER": "输出文件夹",
    "Select output folder…": "选择输出文件夹…",
    "Browse…": "浏览…",
    "Open Folder": "打开文件夹",
    "Enable physical units": "启用物理单位",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "按像素尺寸缩放位移值，并在色条标签显示物理单位。应变为无量纲，不受影响。",
    "/ pixel": "/ 像素",
    "Pixel size": "像素尺寸",
    "fps": "fps",
    "Frame rate": "帧率",
    "Data": "数据",
    "Images": "图像",
    "Animation": "动画",
    "Report": "报告",
    "FORMAT": "格式",
    "NumPy Archive (.npz)": "NumPy 归档 (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV（逐帧）",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ：逐帧一个文件（默认：合并为单个文件）",
    "DISPLACEMENT": "位移",
    "Select:": "选择：",
    "STRAIN": "应变",
    "Run Compute Strain first.": "请先运行“计算应变”。",
    "✓ Parameters file (JSON) always exported": "✓ 参数文件（JSON）始终导出",
    "Export Data": "导出数据",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "批量导入感兴趣区域掩模",
    "Mask Folder:": "掩模文件夹：",
    "(none)": "（无）",
    "Browse...": "浏览…",
    "Available Masks": "可用掩模",
    "Auto-Match by Name": "按文件名自动匹配",
    "Match mask files to frames by number in filename":
        "根据文件名中的数字把掩模文件匹配到对应帧",
    "Assign Sequential": "顺序分配",
    "Assign masks to frames in order starting from frame 0":
        "从第 0 帧起按顺序把掩模分配给各帧",
    "Frame Assignments": "帧分配",
    "Frame": "帧",
    "Image": "图像",
    "Mask": "掩模",
    "Assign Selected ->": "分配所选 ->",
    "Pair selected mask(s) with selected frame(s)":
        "将所选掩模与所选帧配对",
    "Clear All": "全部清除",

    # Canvas area / toolbar
    "Fit": "适配",
    "Fit image to viewport": "将图像适配到视口",
    "100%": "100%",
    "Zoom to 100% (1:1)": "缩放到 100%（1:1）",
    "Zoom in": "放大",
    "–": "–",
    "Zoom out": "缩小",
    "Show Grid": "显示网格",
    "Show/hide computational mesh grid": "显示/隐藏计算网格",
    "Show Subset": "显示子集",
    "Show subset window on hover (requires Grid)": "悬停时显示子集窗口（需要先开启网格）",
    "Placing Starting Points": "正在放置种子点",

    # Color range
    "Range": "范围",
    "Auto": "自动",
    "Min": "最小",
    "Max": "最大",

    # Field selector
    "Disp U": "位移 U",
    "Disp V": "位移 V",

    # Frame navigator / strain navigator
    "Previous frame": "上一帧",
    "Play animation": "播放动画",
    "▶": "▶",
    "Next frame": "下一帧",
    "Playback speed": "播放速度",
    "FRAME 0/0": "第 0/0 帧",
    "⏸": "⏸",
    "Pause animation": "暂停动画",

    # Image list
    "Add": "添加",
    "Edit": "编辑",
    "Need": "待绘",

    # Mesh appearance
    "Mesh color": "网格颜色",
    "Click to choose mesh line color": "点击选择网格线颜色",
    "Line width": "线宽",

    # Physical units widget
    "Use physical units": "使用物理单位",
    "Physical size of one image pixel": "单个图像像素对应的物理尺寸",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)": "采集帧率（用于速度场）",
    "Disp: px  Velocity: px/fr": "位移：px  速度：px/帧",

    # Advanced tuning
    "ADMM Iterations": "ADMM 迭代次数",

    # Strain window
    "Strain Post-Processing": "应变后处理",
    "Compute Strain": "计算应变",
    "Starting…": "启动中…",
    "Complete": "完成",
    "⚠ Params changed -- click Compute Strain": "⚠ 参数已变更 — 请点击“计算应变”",
    "Unit: px/frame": "单位：px/帧",
}


def main() -> None:
    ts_content = TS.read_text(encoding="utf-8")

    miss: list[str] = []
    filled = 0

    def replace_one(match: re.Match) -> str:
        nonlocal filled, miss
        source = match.group(1)
        # Unescape common XML entities lupdate emits.
        key = (source
               .replace("&amp;", "&")
               .replace("&lt;", "<")
               .replace("&gt;", ">")
               .replace("&quot;", '"')
               .replace("&apos;", "'"))
        translation = TRANSLATIONS.get(key)
        if translation is None:
            miss.append(key)
            return match.group(0)
        # Preserve lupdate's XML-escaping in the translation target.
        escaped = (translation
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        filled += 1
        return (f"<source>{source}</source>\n"
                f"        <translation>{escaped}</translation>")

    new_content = re.sub(
        r"<source>(.*?)</source>\s*"
        r"<translation(?: type=\"unfinished\")?>([^<]*)</translation>",
        replace_one,
        ts_content,
        flags=re.DOTALL,
    )

    TS.write_text(new_content, encoding="utf-8")
    print(f"Filled: {filled}  |  Missing from map: {len(miss)}")
    for s in miss[:20]:
        print(f"  MISS: {s!r}")
    if len(miss) > 20:
        print(f"  ... and {len(miss) - 20} more")


if __name__ == "__main__":
    main()
