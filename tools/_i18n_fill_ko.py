"""Fill al_dic_ko.ts with Korean translations.

Terminology conventions (see docs/i18n/glossary.md):
  * DIC technical terms: 서브셋 / 변위 / 변형률 / 메시 / 시드점 /
    기준 프레임 / 적응 세분화
  * Proper nouns kept English: pyALDIC, AL-DIC, IC-GN, ADMM, FFT,
    NCC, FEM, Q8, ROI, POI, RMSE, NumPy, MATLAB, CSV, NPZ, PNG, PDF
  * Placeholders (%1, %2, %n, HTML tags <b>) preserved verbatim
"""

from __future__ import annotations

import re
from pathlib import Path

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_ko.ts"

TRANSLATIONS: dict[str, str] = {
    # MainWindow chrome
    "&Settings": "설정",
    "Language": "언어",
    "Language changed": "언어가 변경되었습니다",
    "Language set to %1. Please restart pyALDIC for all widgets to pick up the new language.":
        "언어가 %1(으)로 설정되었습니다. 모든 화면에 반영하려면 pyALDIC을 재시작하세요.",
    "&File": "파일",
    "Open Session…": "세션 열기…",
    "Save Session…": "세션 저장…",
    "Quit": "종료",
    "Save Session": "세션 저장",
    "Open Session": "세션 열기",
    "Save Session Failed": "세션 저장 실패",
    "Open Session Failed": "세션 열기 실패",
    "pyALDIC Session": "pyALDIC 세션",
    "All Files": "모든 파일",
    "JSON": "JSON",

    # Right sidebar — run controls
    "Run DIC Analysis": "DIC 분석 실행",
    "Cancel": "취소",
    "Cancel the current analysis. Already-computed frames are kept; the run is marked as IDLE (not DONE).":
        "현재 분석을 취소합니다. 이미 계산된 프레임은 유지되며, 실행은 IDLE(DONE 아님) 상태로 표시됩니다.",
    "Export Results": "결과 내보내기",
    "Open Strain Window": "변형률 창 열기",
    "Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.":
        "별도의 후처리 창에서 변형률을 계산·시각화합니다. 완료된 실행의 변위 결과가 필요합니다.",
    "Place at least one Starting Point in each red region before running (red = needs a Starting Point).":
        "실행 전에 각 빨간 영역에 시드점을 하나 이상 배치하세요(빨강 = 시드점 필요).",

    # Progress / Field / Visualization
    "PROGRESS": "진행률",
    "Ready": "준비 완료",
    "ELAPSED  %1": "경과  %1",
    "REMAINING  %1": "남음  %1",
    "%1  —  Frame %2": "%1  —  프레임 %2",
    "FIELD": "표시 필드",
    "Show on deformed frame": "변형 프레임에 표시",
    "When checked, overlay results on the deformed (current) frame instead of the reference frame":
        "체크하면 결과를 기준 프레임 대신 변형(현재) 프레임에 겹쳐 표시합니다",
    "VISUALIZATION": "시각화",
    "Colormap": "색상표",
    "Opacity": "불투명도",
    "Overlay opacity (0 = transparent, 100 = opaque)":
        "오버레이 불투명도(0 = 투명, 100 = 불투명)",
    "PHYSICAL UNITS": "물리 단위",
    "LOG": "로그",
    "Clear": "지우기",

    # Left sidebar
    "IMAGES": "이미지",
    "Drop image folder\nor Browse": "이미지 폴더를 드롭하거나\n찾아보기",
    "Select Image Folder": "이미지 폴더 선택",
    "Natural Sort (1, 2, …, 10)": "자연 정렬 (1, 2, …, 10)",
    "Sort by embedded numbers: image1, image2, …, image10\nDefault (unchecked): lexicographic — best for zero-padded names":
        "파일명 속 숫자 순 정렬: image1, image2, …, image10\n기본(체크 해제): 사전식 — 0 채움 파일명에 적합",
    "WORKFLOW TYPE": "워크플로 유형",
    "INITIAL GUESS": "초기 추정",
    "REGION OF INTEREST": "관심 영역",
    "PARAMETERS": "매개변수",
    "ADVANCED": "고급 설정",

    # Workflow type panel
    "Tracking Mode": "추적 모드",
    "Incremental: each frame is compared to the previous reference frame.\nSuitable for large accumulated deformation, required for large rotations.\n\nAccumulative: every frame is compared to frame 1.\nAccurate for small, monotonic deformation only.":
        "증분형: 각 프레임을 직전 기준 프레임과 비교합니다.\n누적 변형이 큰 경우에 적합하며, 큰 회전에는 필수입니다.\n\n누적형: 각 프레임을 1번 프레임과 비교합니다.\n작고 단조로운 변형에만 정확합니다.",
    "Solver": "솔버",
    "Local DIC: Independent subset matching (IC-GN). Fast,\npreserves sharp local features. Best for small\ndeformations or high-quality images.\n\nAL-DIC: Augmented Lagrangian with global FEM\nregularization. Enforces displacement compatibility\nbetween subsets. Best for large deformations, noisy\nimages, or when strain accuracy matters.":
        "Local DIC: 독립 서브셋 매칭(IC-GN). 빠르고\n국소 특징을 보존합니다. 작은 변형이나 고품질\n이미지에 적합합니다.\n\nAL-DIC: 전역 FEM 정칙화를 갖춘 확장 라그랑주.\n서브셋 간 변위 적합성을 강제합니다. 큰 변형,\n노이즈 이미지, 변형률 정확도가 중요한 경우에 적합합니다.",
    "Reference Update": "기준 프레임 갱신",
    "When the reference frame refreshes during incremental tracking.\nEvery Frame: reset reference every frame (smallest per-step displacement,\nmost robust for large deformation).\nEvery N Frames: reset every N frames (balance speed vs robustness).\nCustom Frames: user-defined list of reference frame indices.":
        "증분 추적 중 기준 프레임 갱신 시점.\n매 프레임: 매 프레임마다 기준 리셋(단계 변위 최소,\n큰 변형에 가장 견고).\nN 프레임마다: N 프레임마다 리셋(속도-견고성 균형).\n사용자 지정: 사용자 정의 기준 프레임 인덱스 목록.",
    "Update reference every N frames": "N 프레임마다 기준 갱신",
    "Interval": "간격",
    "Reference Frames": "기준 프레임",
    "Comma-separated frame indices to use as reference frames (0-based)":
        "기준 프레임으로 사용할 프레임 인덱스(0부터, 쉼표 구분)",

    # ROI toolbar
    "+ Add": "+ 추가",
    "Add region to the Region of Interest (Polygon / Rectangle / Circle)":
        "관심 영역에 도형을 추가합니다(다각형 / 사각형 / 원)",
    "Cut": "잘라내기",
    "Cut region from the Region of Interest (Polygon / Rectangle / Circle)":
        "관심 영역에서 도형을 잘라냅니다(다각형 / 사각형 / 원)",
    "+ Refine": "+ 세분화",
    "Paint extra mesh-refinement zones with a brush\n(only on frame 1 — material points auto-warped to later frames)":
        "브러시로 추가 메시 세분화 영역을 칠합니다\n(프레임 1에서만 — 후속 프레임으로 자동 워프됨)",
    "Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.":
        "세분화 브러시는 프레임 1에서만 사용할 수 있습니다. 프레임 1로 전환해 영역을 칠하세요. 후속 프레임으로 자동 워프됩니다.",
    "Import": "가져오기",
    "Import mask from image file": "이미지 파일에서 마스크 가져오기",
    "Batch Import": "일괄 가져오기",
    "Batch import mask files for multiple frames": "여러 프레임의 마스크 파일을 일괄 가져오기",
    "Save": "저장",
    "Save current mask to PNG file": "현재 마스크를 PNG 파일로 저장",
    "Invert": "반전",
    "Invert the Region of Interest mask": "관심 영역 마스크 반전",
    "Clear all Region of Interest masks": "모든 관심 영역 마스크 지우기",
    "Radius": "반경",
    "Paint": "칠하기",
    "Erase": "지우기",
    "Clear Brush": "브러시 지우기",

    # Parameters panel
    "Subset Size": "서브셋 크기",
    "IC-GN subset window size in pixels (odd number)":
        "IC-GN 서브셋 윈도우 크기(픽셀, 홀수)",
    "Subset Step": "서브셋 간격",
    "Node spacing in pixels (must be power of 2)":
        "노드 간격(픽셀, 2의 거듭제곱이어야 함)",
    "Search Range": "탐색 범위",
    "Initial Seed Search": "초기 시드 탐색",
    "Maximum per-frame displacement the FFT search can detect (pixels).\nSet comfortably larger than the expected inter-frame motion.\nFor large rotations in incremental mode, this must cover\n  radius × sin(per-step angle).":
        "FFT 탐색이 검출할 수 있는 프레임당 최대 변위(픽셀).\n예상 프레임 간 움직임보다 충분히 크게 설정하세요.\n증분 모드의 큰 회전 시 다음을 포함해야 합니다:\n  반경 × sin(단계 각).",
    "Initial half-width (pixels) of the single-point NCC search at each Starting Point.\nAuto-expands 2x per retry if the peak is clipped, up to image half-size.\nOnly affects Starting Point bootstrap; other nodes use F-aware propagation (no per-node search).":
        "각 시드점의 단점 NCC 탐색 초기 반폭(픽셀).\n피크가 잘리면 이미지 절반 크기까지 재시도마다 2배씩 자동 확장합니다.\n시드점 초기화에만 영향을 주며, 다른 노드는 F-aware 전파(노드별 탐색 없음)를 사용합니다.",
    "Refine Inner Boundary": "내부 경계 세분화",
    "Locally refine the mesh along internal mask boundaries\n(holes inside the Region of Interest). Useful for bubble / void edges.":
        "내부 마스크 경계를 따라 메시를 국소적으로 세분화합니다\n(관심 영역 내부의 구멍). 기포/공극 가장자리에 유용합니다.",
    "Refine Outer Boundary": "외부 경계 세분화",
    "Locally refine the mesh along the outer Region of Interest\nboundary.":
        "관심 영역 외부 경계를 따라 메시를 국소적으로 세분화합니다.",
    "Refinement aggressiveness. min element size = max(2, subset_step / 2^level). Applies uniformly to inner-, outer-boundary AND brush-painted refinement zones. Available levels depend on subset size and subset step.":
        "세분화 강도. 최소 요소 크기 = max(2, 서브셋 간격 / 2^레벨). 내부·외부 경계와 브러시로 칠한 영역에 모두 일괄 적용됩니다. 사용 가능한 레벨은 서브셋 크기와 간격에 따라 달라집니다.",
    "Refinement Level": "세분화 레벨",
    "min element size = %1 px  (subset_step=%2, level=%3)":
        "최소 요소 크기 = %1 px  (서브셋 간격=%2, 레벨=%3)",

    # Initial guess widget
    "Starting Points": "시드점",
    "Place a few points; pyALDIC bootstraps each with a single-point NCC and propagates the field along mesh neighbours.\n\nBest for:\n• Large inter-frame displacement (> 50 px)\n• Discontinuous fields (cracks, shear bands)\n• Scenarios where FFT picks wrong peaks\n\nAuto-placed per region when you draw or edit an ROI.":
        "몇 개의 시드점을 배치합니다. pyALDIC은 단점 NCC로 각각을 초기화하고 메시 이웃을 따라 필드를 전파합니다.\n\n적합한 경우:\n• 큰 프레임 간 변위(> 50 px)\n• 불연속 필드(균열, 전단대)\n• FFT가 잘못된 피크를 고르는 경우\n\nROI 생성/편집 시 영역별로 자동 배치됩니다.",
    "Place Starting Points": "시드점 배치",
    "Placing... (click to exit)": "배치 중…(클릭하여 종료)",
    "Enter placement mode on the canvas. Left-click to add, right-click to remove, Esc or click again to exit.":
        "캔버스에서 배치 모드로 들어갑니다. 좌클릭으로 추가, 우클릭으로 제거, Esc 또는 다시 클릭하여 종료합니다.",
    "Auto-place": "자동 배치",
    "Fill empty regions with the highest-NCC node in each. Existing Starting Points are preserved.":
        "각 빈 영역에 최고 NCC 노드를 배치합니다. 기존 시드점은 유지됩니다.",
    "Remove every Starting Point. Faster than right-clicking each one individually.":
        "모든 시드점을 제거합니다. 하나씩 우클릭하는 것보다 빠릅니다.",
    "%1 / %2 regions ready": "%1 / %2 영역 준비 완료",
    "FFT (cross-correlation)": "FFT(상호상관)",
    "Full-grid normalized cross-correlation. Robust within the search radius; the search auto-expands when peaks clip.\n\nBest for:\n• Small-to-moderate smooth motion\n• Well-textured speckle\n• No special user setup needed\n\nCost grows with the search radius, so very large displacements become slow.":
        "전체 격자 정규화 상호상관. 탐색 반경 내에서 견고하며, 피크가 잘리면 탐색이 자동 확장됩니다.\n\n적합한 경우:\n• 작거나 중간 크기의 부드러운 움직임\n• 질감이 좋은 스페클\n• 특별한 설정이 불필요\n\n비용이 탐색 반경과 함께 증가하므로 매우 큰 변위에서는 느려집니다.",
    "Every": "매",
    "Run FFT every N frames. N = 1 means FFT every frame (safest, slowest). N > 1 uses warm-start between resets to limit error propagation to N frames.":
        "N 프레임마다 FFT를 실행합니다. N = 1은 매 프레임 FFT(가장 안전·가장 느림). N > 1은 리셋 사이에 웜스타트를 사용해 오류 전파를 N 프레임 이내로 제한합니다.",
    "(N=1 = every frame)": "(N=1 은 매 프레임)",
    "Only when reference frame updates (incremental only)":
        "기준 프레임 갱신 시에만(증분형만)",
    "Run FFT whenever the reference frame changes; warm-start within each segment. Typical default for incremental mode.":
        "기준 프레임이 바뀔 때마다 FFT를 실행하고, 각 구간 내에서는 웜스타트를 사용합니다. 증분 모드의 표준 기본값입니다.",
    "Previous frame": "이전 프레임",
    "Use the previous frame's converged displacement as the initial guess. No cross-correlation runs.\n\nBest for:\n• Very small inter-frame motion (a few pixels)\n• Fastest option when motion is smooth\n\nErrors can accumulate over long sequences. Prefer FFT or Starting Points on noisy data or when motion is larger.":
        "이전 프레임의 수렴된 변위를 초기 추정으로 사용합니다. 상호상관을 실행하지 않습니다.\n\n적합한 경우:\n• 매우 작은 프레임 간 움직임(몇 픽셀)\n• 움직임이 부드러울 때 가장 빠른 옵션\n\n긴 시퀀스에서 오류가 누적될 수 있습니다. 노이즈 데이터나 움직임이 클 때는 FFT 또는 시드점을 권장합니다.",

    # ROI hint
    "Load images first, then draw a Region of Interest on frame 1.":
        "먼저 이미지를 불러온 후, 프레임 1에 관심 영역을 그리세요.",
    "<b>Accumulative mode</b> — only frame 1 needs a Region of Interest. All later frames are compared against it directly.":
        "<b>누적 모드</b> — 관심 영역은 프레임 1에만 필요합니다. 이후 프레임은 모두 직접 비교됩니다.",
    "<b>Incremental, every frame</b> — frame 1 needs a Region of Interest. It is automatically warped forward to each later frame (no per-frame drawing required).":
        "<b>증분, 매 프레임</b> — 프레임 1에 관심 영역이 필요합니다. 이후 프레임으로 자동 전진 워프됩니다(프레임별 그리기 불필요).",
    "<b>Incremental, every %1 frames</b> — draw a Region of Interest on frames: <b>%2</b> (%3 reference frames total).":
        "<b>증분, %1 프레임마다</b> — 다음 프레임에 관심 영역을 그리세요: <b>%2</b>(기준 프레임 총 %3개).",
    "<b>Incremental, custom</b> — no custom reference frames set. Frame 1 will be the only reference; add more indices in the Reference Frames field.":
        "<b>증분, 사용자 지정</b> — 사용자 정의 기준 프레임이 설정되지 않았습니다. 프레임 1이 유일한 기준이 됩니다. 기준 프레임 입력란에 인덱스를 추가하세요.",
    "<b>Incremental, custom</b> — draw a Region of Interest on frames: <b>%1</b> (%2 reference frames total).":
        "<b>증분, 사용자 지정</b> — 다음 프레임에 관심 영역을 그리세요: <b>%1</b>(기준 프레임 총 %2개).",
    "Draw a Region of Interest on frame 1.": "프레임 1에 관심 영역을 그리세요.",

    # Export dialog
    "All": "모두 선택",
    "None": "모두 해제",
    "OUTPUT FOLDER": "출력 폴더",
    "Select output folder…": "출력 폴더 선택…",
    "Browse…": "찾아보기…",
    "Open Folder": "폴더 열기",
    "Enable physical units": "물리 단위 활성화",
    "Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.":
        "변위 값을 픽셀 크기로 스케일링하고 색상 막대 레이블에 물리 단위를 표시합니다. 변형률은 무차원이므로 영향받지 않습니다.",
    "/ pixel": "/ 픽셀",
    "Pixel size": "픽셀 크기",
    "fps": "fps",
    "Frame rate": "프레임 속도",
    "Data": "데이터",
    "Images": "이미지",
    "Animation": "애니메이션",
    "Report": "보고서",
    "FORMAT": "형식",
    "NumPy Archive (.npz)": "NumPy 아카이브 (.npz)",
    "MATLAB (.mat)": "MATLAB (.mat)",
    "CSV (per frame)": "CSV(프레임별)",
    "NPZ: one file per frame (default: single merged file)":
        "NPZ: 프레임별 1 파일(기본값: 통합 단일 파일)",
    "DISPLACEMENT": "변위",
    "Select:": "선택:",
    "STRAIN": "변형률",
    "Run Compute Strain first.": "먼저 「변형률 계산」을 실행하세요.",
    "✓ Parameters file (JSON) always exported": "✓ 매개변수 파일(JSON)은 항상 내보내집니다",
    "Export Data": "데이터 내보내기",

    # Batch import dialog
    "Batch Import Region of Interest Masks": "관심 영역 마스크 일괄 가져오기",
    "Mask Folder:": "마스크 폴더:",
    "(none)": "(없음)",
    "Browse...": "찾아보기…",
    "Available Masks": "사용 가능한 마스크",
    "Auto-Match by Name": "이름으로 자동 매칭",
    "Match mask files to frames by number in filename":
        "파일명의 숫자로 마스크 파일을 프레임에 매칭합니다",
    "Assign Sequential": "순차 할당",
    "Assign masks to frames in order starting from frame 0":
        "프레임 0부터 순서대로 마스크를 프레임에 할당합니다",
    "Frame Assignments": "프레임 할당",
    "Frame": "프레임",
    "Image": "이미지",
    "Mask": "마스크",
    "Assign Selected ->": "선택 항목 할당 ->",
    "Pair selected mask(s) with selected frame(s)":
        "선택한 마스크와 선택한 프레임을 짝지웁니다",
    "Clear All": "모두 지우기",

    # Canvas area / toolbar
    "Fit": "맞춤",
    "Fit image to viewport": "이미지를 뷰포트에 맞춤",
    "100%": "100%",
    "Zoom to 100% (1:1)": "100%(1:1) 확대",
    "Zoom in": "확대",
    "–": "–",
    "Zoom out": "축소",
    "Show Grid": "격자 표시",
    "Show/hide computational mesh grid": "계산 메시 격자 표시/숨김",
    "Show Subset": "서브셋 표시",
    "Show subset window on hover (requires Grid)":
        "마우스 오버 시 서브셋 창 표시(격자 필요)",
    "Placing Starting Points": "시드점 배치 중",

    # Color range
    "Range": "범위",
    "Auto": "자동",
    "Min": "최소",
    "Max": "최대",

    # Field selector
    "Disp U": "변위 U",
    "Disp V": "변위 V",

    # Frame navigator / strain navigator
    "Play animation": "애니메이션 재생",
    "▶": "▶",
    "Next frame": "다음 프레임",
    "Playback speed": "재생 속도",
    "FRAME 0/0": "프레임 0/0",
    "⏸": "⏸",
    "Pause animation": "애니메이션 일시정지",

    # Image list
    "Add": "추가",
    "Edit": "편집",
    "Need": "필요",

    # Mesh appearance
    "Mesh color": "메시 색상",
    "Click to choose mesh line color": "메시 선 색상 선택을 위해 클릭",
    "Line width": "선 너비",

    # Physical units widget
    "Use physical units": "물리 단위 사용",
    "Physical size of one image pixel": "이미지 픽셀 1개의 물리 크기",
    "/ px": "/ px",
    "Acquisition frame rate (used for velocity field)":
        "획득 프레임 속도(속도장에 사용)",
    "Disp: px  Velocity: px/fr": "변위: px  속도: px/fr",

    # Advanced tuning
    "ADMM Iterations": "ADMM 반복 횟수",
    "Number of ADMM alternating minimization cycles for AL-DIC.\n1 = single global pass (fastest), 3 = default,\n5+ = diminishing returns for most cases.":
        "AL-DIC의 ADMM 교대 최소화 반복 횟수.\n1 = 단일 전역 패스(가장 빠름), 3 = 기본값,\n5 이상은 대부분의 경우 수확 체감.",
    "Only affects AL-DIC solver. Ignored by Local DIC.":
        "AL-DIC 솔버에만 적용됩니다. Local DIC에서는 무시됩니다.",
    "Auto-expand FFT search on clipped peaks":
        "피크가 잘리면 FFT 탐색 자동 확장",
    "When the NCC peak reaches the edge of the search region, automatically retry with a larger region (up to image half-size, 6 retries with 2x growth).\n\nOnly relevant for the FFT init-guess mode.":
        "NCC 피크가 탐색 영역 경계에 도달하면 더 넓은 영역으로 자동 재시도합니다(이미지 절반 크기까지, 2배씩 6회).\n\nFFT 초기 추정 모드에만 관련됩니다.",

    # Canvas overlay
    "Mode": "모드",
    "Init": "초기값",
    "Accumulative": "누적형",
    "Incremental": "증분형",
    "Local DIC": "Local DIC",
    "Every Frame": "매 프레임",
    "Every N Frames": "N 프레임마다",
    "Custom Frames": "사용자 지정 프레임",
    "ADMM (%1 iter)": "ADMM(%1회 반복)",
    "FFT every frame": "매 프레임 FFT",
    "FFT every %1 fr": "%1 프레임마다 FFT",
    "FFT": "FFT",

    # Log messages
    "Load images first.": "먼저 이미지를 불러오세요.",
    "Building pipeline configuration...": "파이프라인 설정 구성 중…",
    "Loading images...": "이미지 불러오는 중…",

    # Strain window
    "Strain Post-Processing": "변형률 후처리",
    "Compute Strain": "변형률 계산",
    "Starting…": "시작 중…",
    "Complete": "완료",
    "⚠ Params changed -- click Compute Strain": "⚠ 매개변수가 변경됨 — 「변형률 계산」을 클릭하세요",
    "Unit: px/frame": "단위: px/frame",

    # Strain field selector
    "STRAIN PARAMETERS": "변형률 매개변수",

    # Strain param panel
    "Method": "방법",
    "Plane fitting": "평면 피팅",
    "FEM nodal": "FEM 절점",
    "VSG size": "VSG 크기",
    "Strain field smoothing": "변형률장 평활화",
    "Strain type": "변형률 종류",
    "Infinitesimal": "미소 변형률",
    "Eulerian": "오일러 변형률",
    "Green-Lagrangian": "그린-라그랑주 변형률",
    "Gaussian smoothing of the strain field after computation.\nσ is the Gaussian kernel width; 'step' = DIC node spacing.\n  Light  (0.5 × step):  subtle, preserves fine features.\n  Medium (1 × step):    balanced, recommended for noisy data.\n  Strong (2 × step) ⚠:  aggressive, may blur real gradients.":
        "계산 후 변형률장에 가우스 평활화를 적용합니다.\nσ는 가우스 커널 너비, 'step'은 DIC 노드 간격입니다.\n  Light  (0.5 × step):  약함, 세부를 보존.\n  Medium (1 × step):    균형, 노이즈 데이터에 권장.\n  Strong (2 × step) ⚠: 강함, 실제 기울기를 흐릴 수 있음.",

    # Strain window export tooltip
    "Export displacement and strain results to NPZ / MAT / CSV / PNG":
        "변위 및 변형률 결과를 NPZ / MAT / CSV / PNG로 내보내기",
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
    print(f"[ko] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
