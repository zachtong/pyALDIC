"""One-shot helper: fill the current batch of unfinished translations.

Usage:
    python tools/fill_translations.py

Reads src/al_dic/i18n/source/al_dic_<lang>.ts for every language in
LANGUAGES, finds each <translation type="unfinished">…</translation>
entry whose <source> matches a key in TRANSLATIONS[lang], and rewrites
it as a finished <translation>.

After running, invoke `python tools/i18n.py compile` to rebuild the
.qm runtime catalogs.

This script exists because we batch-translate a round of strings and
want them committed together, not as interactive drive-bys from inside
Qt Linguist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import unescape as _xml_unescape

# pyside6-lupdate serialises .ts text with the full set of five XML
# predefined entities, i.e. it also escapes ' -> &apos; and " -> &quot;.
# Stdlib escape()/unescape() only handle & < > by default, so we extend
# both directions to stay byte-identical to lupdate. Otherwise any
# translation containing an apostrophe (very common in French: d'abord,
# d'intérêt, l'image, ...) drifts on the next `i18n.py extract` and the
# CI Gate A "extract drift" check fails.
_XML_EXTRA_ESCAPE = {"'": "&apos;", '"': "&quot;"}
_XML_EXTRA_UNESCAPE = {"&apos;": "'", "&quot;": '"'}


def escape(text: str) -> str:
    """XML-escape matching pyside6-lupdate (includes &apos; and &quot;)."""
    return _xml_escape(text, _XML_EXTRA_ESCAPE)


def unescape(text: str) -> str:
    """Inverse of escape(); also resolves &apos; and &quot; so translation
    keys containing curly quotes match our Python dict keys.
    """
    return _xml_unescape(text, _XML_EXTRA_UNESCAPE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_DIR = PROJECT_ROOT / "src" / "al_dic" / "i18n" / "source"

LANGUAGES = ("zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es")


# Translations keyed by English source string.
# Each inner dict maps language code -> translated string.
# Keep \n, %1, %2, %n placeholders literal.
#
# Short words / common UI labels
TRANSLATIONS: dict[str, dict[str, str]] = {
    # ===== MainWindow — ROI toolbar feedback after reopening a session ======
    "Frame %1 has no Region of Interest of its own — frame 1's is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.": {
        "zh_CN": "帧 %1 没有自己的感兴趣区域 — 计算时使用帧 1 的感兴趣区域。请切换到帧 1 编辑，或导入掩码为此帧单独指定。",
        "zh_TW": "幀 %1 沒有自己的感興趣區域 — 計算時使用幀 1 的感興趣區域。請切換到幀 1 編輯，或匯入遮罩為此幀單獨指定。",
        "ja":    "フレーム %1 には独自の関心領域がありません — 計算にはフレーム 1 の関心領域を使用します。フレーム 1 に切り替えて編集するか、マスクをインポートしてこのフレーム専用の領域を設定してください。",
        "ko":    "프레임 %1에는 자체 관심 영역이 없습니다 — 계산에는 프레임 1의 관심 영역을 사용합니다. 프레임 1로 전환하여 편집하거나 마스크를 가져와 이 프레임 전용으로 지정하세요.",
        "de":    "Bild %1 hat keine eigene Region of Interest — für die Berechnung wird die von Bild 1 verwendet. Wechseln Sie zu Bild 1, um sie zu bearbeiten, oder importieren Sie eine Maske, um diesem Bild eine eigene zu geben.",
        "fr":    "L'image %1 n'a pas de région d'intérêt propre — celle de l'image 1 est utilisée pour le calcul. Passez à l'image 1 pour la modifier, ou importez un masque pour donner à cette image la sienne.",
        "es":    "El fotograma %1 no tiene una región de interés propia — se usa la del fotograma 1 para el cálculo. Cambie al fotograma 1 para editarla, o importe una máscara para dar a este fotograma la suya.",
    },
    "No Region of Interest to save — load images first.": {
        "zh_CN": "没有可保存的感兴趣区域 — 请先加载图像。",
        "zh_TW": "沒有可儲存的感興趣區域 — 請先載入影像。",
        "ja":    "保存できる関心領域がありません — 先に画像を読み込んでください。",
        "ko":    "저장할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.",
        "de":    "Keine Region of Interest zum Speichern — laden Sie zuerst Bilder.",
        "fr":    "Aucune région d'intérêt à enregistrer — chargez d'abord des images.",
        "es":    "No hay región de interés que guardar — cargue primero las imágenes.",
    },
    "No Region of Interest to invert — load images first.": {
        "zh_CN": "没有可反选的感兴趣区域 — 请先加载图像。",
        "zh_TW": "沒有可反選的感興趣區域 — 請先載入影像。",
        "ja":    "反転できる関心領域がありません — 先に画像を読み込んでください。",
        "ko":    "반전할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.",
        "de":    "Keine Region of Interest zum Invertieren — laden Sie zuerst Bilder.",
        "fr":    "Aucune région d'intérêt à inverser — chargez d'abord des images.",
        "es":    "No hay región de interés que invertir — cargue primero las imágenes.",
    },
    "Region of Interest mask is empty.": {
        "zh_CN": "感兴趣区域掩码为空。",
        "zh_TW": "感興趣區域遮罩為空。",
        "ja":    "関心領域のマスクが空です。",
        "ko":    "관심 영역 마스크가 비어 있습니다.",
        "de":    "Die Region-of-Interest-Maske ist leer.",
        "fr":    "Le masque de la région d'intérêt est vide.",
        "es":    "La máscara de la región de interés está vacía.",
    },
    "Define a Region of Interest on frame 1 first.": {
        "zh_CN": "请先在帧 1 上定义感兴趣区域。",
        "zh_TW": "請先在幀 1 上定義感興趣區域。",
        "ja":    "まずフレーム 1 で関心領域を定義してください。",
        "ko":    "먼저 프레임 1에서 관심 영역을 정의하세요.",
        "de":    "Definieren Sie zuerst eine Region of Interest auf Bild 1.",
        "fr":    "Définissez d'abord une région d'intérêt sur l'image 1.",
        "es":    "Defina primero una región de interés en el fotograma 1.",
    },
    "Mask saved to %1": {
        "zh_CN": "掩码已保存至 %1",
        "zh_TW": "遮罩已儲存至 %1",
        "ja":    "マスクを %1 に保存しました",
        "ko":    "마스크를 %1에 저장했습니다",
        "de":    "Maske gespeichert unter %1",
        "fr":    "Masque enregistré dans %1",
        "es":    "Máscara guardada en %1",
    },
    # ========== MainWindow — relink moved session images (all 8 locales) ======
    "Locate Session Images": {
        "zh_CN": "定位会话图片",
        "zh_TW": "定位工作階段圖片",
        "ja":    "セッション画像の場所を指定",
        "ko":    "세션 이미지 위치 지정",
        "de":    "Sitzungsbilder suchen",
        "fr":    "Localiser les images de la session",
        "es":    "Localizar imágenes de la sesión",
    },
    "Select Image Folder": {
        "zh_CN": "选择图像文件夹",
        "zh_TW": "選擇影像資料夾",
        "ja":    "画像フォルダを選択",
        "ko":    "이미지 폴더 선택",
        "de":    "Bildordner auswählen",
        "fr":    "Sélectionner le dossier d'images",
        "es":    "Seleccionar carpeta de imágenes",
    },
    "The image folder saved with this session was not found:\n%1\n\nResults were restored. To show the background images, select the folder that now contains them.": {
        "zh_CN": "未找到此会话保存的图片文件夹:\n%1\n\n结果已恢复。要显示背景图片,请选择现在包含这些图片的文件夹。",
        "zh_TW": "找不到此工作階段儲存的圖片資料夾:\n%1\n\n結果已還原。若要顯示背景圖片,請選擇現在包含這些圖片的資料夾。",
        "ja":    "このセッションに保存された画像フォルダが見つかりませんでした:\n%1\n\n結果は復元されました。背景画像を表示するには、現在それらが入っているフォルダを選択してください。",
        "ko":    "이 세션에 저장된 이미지 폴더를 찾을 수 없습니다:\n%1\n\n결과는 복원되었습니다. 배경 이미지를 표시하려면 현재 이미지가 들어 있는 폴더를 선택하세요.",
        "de":    "Der mit dieser Sitzung gespeicherte Bildordner wurde nicht gefunden:\n%1\n\nDie Ergebnisse wurden wiederhergestellt. Wählen Sie den Ordner, der die Bilder jetzt enthält, um den Hintergrund anzuzeigen.",
        "fr":    "Le dossier d'images enregistré avec cette session est introuvable :\n%1\n\nLes résultats ont été restaurés. Pour afficher les images d'arrière-plan, sélectionnez le dossier qui les contient désormais.",
        "es":    "No se encontró la carpeta de imágenes guardada con esta sesión:\n%1\n\nLos resultados se restauraron. Para mostrar las imágenes de fondo, seleccione la carpeta que ahora las contiene.",
    },
    # ========== StrainWindow — cancel strain computation (all 8 locales) ======
    "Cancel": {
        "zh_CN": "取消",
        "zh_TW": "取消",
        "ja":    "キャンセル",
        "ko":    "취소",
        "de":    "Abbrechen",
        "fr":    "Annuler",
        "es":    "Cancelar",
    },
    "Cancelling…": {
        "zh_CN": "正在取消…",
        "zh_TW": "正在取消…",
        "ja":    "キャンセル中…",
        "ko":    "취소 중…",
        "de":    "Wird abgebrochen…",
        "fr":    "Annulation…",
        "es":    "Cancelando…",
    },
    "Cancel the running strain computation. The previous strain result is kept.": {
        "zh_CN": "取消正在进行的应变计算。保留之前的应变结果。",
        "zh_TW": "取消正在進行的應變計算。保留先前的應變結果。",
        "ja":    "実行中のひずみ計算をキャンセルします。以前のひずみ結果は保持されます。",
        "ko":    "실행 중인 변형률 계산을 취소합니다. 이전 변형률 결과는 유지됩니다.",
        "de":    "Laufende Dehnungsberechnung abbrechen. Das vorherige Dehnungsergebnis bleibt erhalten.",
        "fr":    "Annuler le calcul de déformation en cours. Le résultat de déformation précédent est conservé.",
        "es":    "Cancelar el cálculo de deformación en curso. Se conserva el resultado de deformación anterior.",
    },
    "Strain computation cancelled.": {
        "zh_CN": "应变计算已取消。",
        "zh_TW": "應變計算已取消。",
        "ja":    "ひずみ計算をキャンセルしました。",
        "ko":    "변형률 계산 취소됨.",
        "de":    "Dehnungsberechnung abgebrochen.",
        "fr":    "Calcul de déformation annulé.",
        "es":    "Cálculo de deformación cancelado.",
    },
    # ========== PipelineController — Cancel button tooltip (all 8 locales) ======
    "Cancel the current analysis. Frames already computed are kept "
    "so you can review or export the partial run.": {
        "zh_CN": "取消当前分析。已计算的帧会被保留，你可以查看或导出这部分结果。",
        "zh_TW": "取消當前分析。已計算的影格會被保留，你可以檢視或匯出這部分結果。",
        "ja":    "現在の解析をキャンセルします。計算済みのフレームは保持され、"
                 "途中までの結果を確認またはエクスポートできます。",
        "ko":    "현재 분석을 취소합니다. 이미 계산된 프레임은 유지되므로 부분 "
                 "결과를 검토하거나 내보낼 수 있습니다.",
        "de":    "Aktuelle Analyse abbrechen. Bereits berechnete Bilder bleiben "
                 "erhalten, sodass Sie den Teillauf ansehen oder exportieren "
                 "können.",
        "fr":    "Annuler l'analyse en cours. Les images déjà calculées sont "
                 "conservées, ce qui permet de consulter ou d'exporter le "
                 "résultat partiel.",
        "es":    "Cancelar el análisis actual. Los fotogramas ya calculados se "
                 "conservan, por lo que puede revisar o exportar la ejecución "
                 "parcial.",
    },
    # ========== StrainParamPanel — edge trim (all 8 locales) ==========
    "Trim low-confidence edges": {
        "zh_CN": "裁剪低置信度边缘",
        "zh_TW": "裁剪低可信度邊緣",
        "ja":    "低信頼度のエッジを除去",
        "ko":    "저신뢰도 가장자리 잘라내기",
        "de":    "Ränder mit geringer Konfidenz beschneiden",
        "fr":    "Rogner les bords peu fiables",
        "es":    "Recortar bordes de baja confianza",
    },
    "Hides low-confidence strain at ROI / hole edges, where the VSG "
    "window crosses the boundary and the local plane fit becomes "
    "one-sided and unreliable.\n\n"
    "• Coefficient × VSG radius = width of the trimmed boundary band.\n"
    "• 0.00 = keep every node (no trimming).\n"
    "• 0.70 = recommended (trims where edge error rises sharply).\n"
    "• 1.00 = strictest (trim any node whose window touches the edge).\n\n"
    "Only applies when Method = Plane fitting.": {
        "zh_CN": "在 ROI / 孔洞边缘隐藏低置信度的应变：那里 VSG 窗口跨越边界，"
                 "局部平面拟合变成单边、不可靠。\n\n"
                 "• 系数 × VSG 半径 = 裁剪边界带的宽度。\n"
                 "• 0.00 = 保留所有节点（不裁剪）。\n"
                 "• 0.70 = 推荐（裁掉误差明显上升的区域）。\n"
                 "• 1.00 = 最严格（窗口一旦触及边界即裁剪）。\n\n"
                 "仅在 方法 = 平面拟合 时生效。",
        "zh_TW": "在 ROI / 孔洞邊緣隱藏低可信度的應變：那裡 VSG 視窗跨越邊界，"
                 "局部平面擬合變成單邊、不可靠。\n\n"
                 "• 係數 × VSG 半徑 = 裁剪邊界帶的寬度。\n"
                 "• 0.00 = 保留所有節點（不裁剪）。\n"
                 "• 0.70 = 推薦（裁掉誤差明顯上升的區域）。\n"
                 "• 1.00 = 最嚴格（視窗一旦觸及邊界即裁剪）。\n\n"
                 "僅在 方法 = 平面擬合 時生效。",
        "ja":    "ROI / 穴の縁で、VSG ウィンドウが境界をまたぎ、局所的な平面"
                 "フィッティングが片側的かつ不正確になる箇所の、低信頼度のひずみ"
                 "を非表示にします。\n\n"
                 "• 係数 × VSG 半径 = トリミングされる境界帯の幅。\n"
                 "• 0.00 = すべてのノードを保持（トリミングなし）。\n"
                 "• 0.70 = 推奨（縁の誤差が急増する箇所をトリミング）。\n"
                 "• 1.00 = 最も厳格（ウィンドウが縁に触れるノードをすべて"
                 "トリミング）。\n\n"
                 "方法 = 平面フィッティング の場合のみ有効です。",
        "ko":    "ROI / 구멍 가장자리에서 VSG 창이 경계를 넘어 국소 평면 피팅이 "
                 "한쪽으로 치우쳐 신뢰할 수 없게 되는 부분의 저신뢰도 변형률을 "
                 "숨깁니다.\n\n"
                 "• 계수 × VSG 반경 = 잘라내는 경계 띠의 폭.\n"
                 "• 0.00 = 모든 노드 유지(잘라내기 없음).\n"
                 "• 0.70 = 권장(가장자리 오차가 급증하는 곳을 잘라냄).\n"
                 "• 1.00 = 가장 엄격(창이 가장자리에 닿는 모든 노드를 잘라냄).\n\n"
                 "Method = 평면 피팅 일 때만 적용됩니다.",
        "de":    "Blendet Dehnung mit geringer Konfidenz an ROI-/Loch-Rändern "
                 "aus, wo das VSG-Fenster die Grenze überschreitet und die "
                 "lokale Ebenenanpassung einseitig und unzuverlässig wird.\n\n"
                 "• Koeffizient × VSG-Radius = Breite des beschnittenen "
                 "Randbereichs.\n"
                 "• 0.00 = jeden Knoten behalten (kein Beschneiden).\n"
                 "• 0.70 = empfohlen (beschneidet, wo der Randfehler stark "
                 "ansteigt).\n"
                 "• 1.00 = strengste Einstellung (beschneidet jeden Knoten, "
                 "dessen Fenster den Rand berührt).\n\n"
                 "Gilt nur bei Methode = Ebenenanpassung.",
        "fr":    "Masque la déformation peu fiable aux bords de la ROI / des "
                 "trous, là où la fenêtre VSG franchit la frontière et où "
                 "l'ajustement de plan local devient unilatéral et peu "
                 "fiable.\n\n"
                 "• Coefficient × rayon VSG = largeur de la bande de bord "
                 "rognée.\n"
                 "• 0.00 = conserver tous les nœuds (aucun rognage).\n"
                 "• 0.70 = recommandé (rogne là où l'erreur de bord augmente "
                 "fortement).\n"
                 "• 1.00 = le plus strict (rogne tout nœud dont la fenêtre "
                 "touche le bord).\n\n"
                 "Ne s'applique que lorsque Méthode = Ajustement de plan.",
        "es":    "Oculta la deformación de baja confianza en los bordes de la "
                 "ROI / huecos, donde la ventana VSG cruza el límite y el "
                 "ajuste de plano local se vuelve unilateral y poco fiable.\n\n"
                 "• Coeficiente × radio VSG = ancho de la banda de borde "
                 "recortada.\n"
                 "• 0.00 = conservar todos los nodos (sin recorte).\n"
                 "• 0.70 = recomendado (recorta donde el error de borde "
                 "aumenta bruscamente).\n"
                 "• 1.00 = más estricto (recorta cualquier nodo cuya ventana "
                 "toque el borde).\n\n"
                 "Solo se aplica cuando Método = Ajuste de plano.",
    },
    "Trimmed: %1 nodes (%2%)": {
        "zh_CN": "已裁剪：%1 个节点 (%2%)",
        "zh_TW": "已裁剪：%1 個節點 (%2%)",
        "ja":    "トリミング: %1 ノード (%2%)",
        "ko":    "잘라냄: 노드 %1개 (%2%)",
        "de":    "Beschnitten: %1 Knoten (%2%)",
        "fr":    "Rognés : %1 nœuds (%2%)",
        "es":    "Recortados: %1 nodos (%2%)",
    },
    "Strain window ≈ %1×%2 nodes": {
        "zh_CN": "应变窗口 ≈ %1×%2 节点",
        "zh_TW": "應變視窗 ≈ %1×%2 節點",
        "ja":    "ひずみウィンドウ ≈ %1×%2 ノード",
        "ko":    "변형률 윈도우 ≈ %1×%2 노드",
        "de":    "Dehnungsfenster ≈ %1×%2 Knoten",
        "fr":    "Fenêtre de déformation ≈ %1×%2 nœuds",
        "es":    "Ventana de deformación ≈ %1×%2 nodos",
    },
    "Number of mesh nodes per axis inside the circular VSG window on a "
    "uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane "
    "fit uses every node within the radius; on a refined mesh the count "
    "varies locally.": {
        "zh_CN": "均匀网格下圆形 VSG 窗口内每个轴向的网格节点数："
                 "2 × floor(VSG 半径 / 节点间距) + 1。平面拟合使用半径内的"
                 "所有节点；在加密网格上该数量会局部变化。",
        "zh_TW": "均勻網格下圓形 VSG 視窗內每個軸向的網格節點數："
                 "2 × floor(VSG 半徑 / 節點間距) + 1。平面擬合使用半徑內的"
                 "所有節點；在加密網格上該數量會局部變化。",
        "ja":    "均一メッシュにおける円形 VSG ウィンドウ内の軸ごとの"
                 "メッシュノード数：2 × floor(VSG 半径 / ノード間隔) + 1。"
                 "平面フィットは半径内のすべてのノードを使用します。"
                 "細分化メッシュではこの数は局所的に変化します。",
        "ko":    "균일 메시에서 원형 VSG 윈도우 내 축당 메시 노드 수: "
                 "2 × floor(VSG 반경 / 노드 간격) + 1. 평면 피팅은 반경 내 "
                 "모든 노드를 사용하며, 세분화된 메시에서는 이 수가 "
                 "국소적으로 달라집니다.",
        "de":    "Anzahl der Netzknoten pro Achse innerhalb des kreisförmigen "
                 "VSG-Fensters bei gleichmäßigem Netz: 2 × floor(VSG-Radius / "
                 "Knotenabstand) + 1. Die Ebenenanpassung verwendet jeden "
                 "Knoten innerhalb des Radius; bei verfeinertem Netz variiert "
                 "die Anzahl lokal.",
        "fr":    "Nombre de nœuds de maillage par axe à l'intérieur de la "
                 "fenêtre VSG circulaire sur un maillage uniforme : 2 × "
                 "floor(rayon VSG / espacement des nœuds) + 1. L'ajustement de "
                 "plan utilise tous les nœuds dans le rayon ; sur un maillage "
                 "raffiné, ce nombre varie localement.",
        "es":    "Número de nodos de malla por eje dentro de la ventana VSG "
                 "circular en una malla uniforme: 2 × floor(radio VSG / "
                 "espaciado de nodos) + 1. El ajuste de plano usa todos los "
                 "nodos dentro del radio; en una malla refinada, ese número "
                 "varía localmente.",
    },

    # ========== ROI drawing — 3-point circle (all 8 locales) ==========
    "Circle (3-point)": {
        "zh_CN": "圆（三点）", "zh_TW": "圓（三點）",
        "ja": "円（3 点）", "ko": "원(3점)",
        "de": "Kreis (3 Punkte)", "fr": "Cercle (3 points)",
        "es": "Círculo (3 puntos)",
    },
    "Load images first before drawing a Region of Interest.": {
        "zh_CN": "请先加载图像，再绘制感兴趣区域。",
        "zh_TW": "請先載入影像，再繪製感興趣區域。",
        "ja": "関心領域を描画する前に、まず画像を読み込んでください。",
        "ko": "관심 영역을 그리기 전에 먼저 이미지를 불러오세요.",
        "de": "Laden Sie zuerst Bilder, bevor Sie eine Region of Interest zeichnen.",
        "fr": "Chargez d'abord des images avant de dessiner une région d'intérêt.",
        "es": "Cargue primero las imágenes antes de dibujar una región de interés.",
    },
    "The three points are nearly collinear — pick points spread around the circle's edge.": {
        "zh_CN": "三个点几乎共线 — 请在圆周上分散地选取三个点。",
        "zh_TW": "三個點幾乎共線 — 請在圓周上分散地選取三個點。",
        "ja": "3 点がほぼ一直線です — 円周上に分散させて 3 点を選んでください。",
        "ko": "세 점이 거의 일직선입니다 — 원의 가장자리에 고르게 세 점을 찍으세요.",
        "de": "Die drei Punkte sind fast kollinear — wählen Sie Punkte, die über den Kreisrand verteilt sind.",
        "fr": "Les trois points sont presque colinéaires — choisissez des points répartis sur le bord du cercle.",
        "es": "Los tres puntos son casi colineales — elija puntos repartidos por el borde del círculo.",
    },

    # ========== Batch-import dialog backlog (zh_CN already done; fill 6 others) ==========
    "Select Mask Folder": {
        "zh_TW": "選擇遮罩資料夾", "ja": "マスクフォルダを選択",
        "ko": "마스크 폴더 선택", "de": "Maskenordner auswählen",
        "fr": "Sélectionner le dossier de masques",
        "es": "Seleccionar carpeta de máscaras",
    },
    "Failed to read mask file.": {
        "zh_TW": "無法讀取遮罩檔案。", "ja": "マスクファイルの読み込みに失敗しました。",
        "ko": "마스크 파일을 읽지 못했습니다.",
        "de": "Maskendatei konnte nicht gelesen werden.",
        "fr": "Échec de la lecture du fichier de masque.",
        "es": "No se pudo leer el archivo de máscara.",
    },
    "Mismatched shape: %1×%2 (expected %3×%4)": {
        "zh_TW": "尺寸不符：%1×%2（預期 %3×%4）",
        "ja": "形状が一致しません: %1×%2 (期待値 %3×%4)",
        "ko": "형상 불일치: %1×%2 (예상 %3×%4)",
        "de": "Größe stimmt nicht: %1×%2 (erwartet %3×%4)",
        "fr": "Forme incompatible : %1×%2 (attendu %3×%4)",
        "es": "Forma no coincide: %1×%2 (se esperaba %3×%4)",
    },
    "%n mask(s) have mismatched sizes and are disabled.": {
        "zh_TW": "%n 個遮罩尺寸不符，已停用。",
        "ja": "%n 個のマスクはサイズが一致しないため無効化されました。",
        "ko": "크기가 일치하지 않는 마스크 %n개가 비활성화되었습니다.",
        "de": "%n Maske(n) haben abweichende Größen und sind deaktiviert.",
        "fr": "%n masque(s) ont des tailles incompatibles et sont désactivés.",
        "es": "%n máscara(s) tienen tamaños no coincidentes y están deshabilitadas.",
    },
    "Invalid assignment": {
        "zh_TW": "無效的指派", "ja": "無効な割り当て", "ko": "잘못된 할당",
        "de": "Ungültige Zuordnung", "fr": "Affectation non valide",
        "es": "Asignación no válida",
    },
    "A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.": {
        "zh_TW": "一個影格只能對應一個遮罩。請選擇恰好一個遮罩，或選擇多個影格以將同一個遮罩套用到多個影格。",
        "ja": "1 つのフレームに割り当てられるマスクは 1 つだけです。マスクを 1 つだけ選択するか、複数のフレームを選択して 1 つのマスクを複数に割り当ててください。",
        "ko": "한 프레임에는 마스크를 하나만 지정할 수 있습니다. 마스크를 정확히 하나 선택하거나, 여러 프레임을 선택해 하나의 마스크를 여러 프레임에 지정하세요.",
        "de": "Ein Bild kann nur eine Maske haben. Wählen Sie genau eine Maske aus oder wählen Sie mehrere Bilder, um eine Maske mehreren zuzuweisen.",
        "fr": "Une image ne peut avoir qu'un seul masque. Sélectionnez exactement un masque, ou sélectionnez plusieurs images pour attribuer un masque à plusieurs.",
        "es": "Un fotograma solo puede tener una máscara. Seleccione exactamente una máscara, o seleccione varios fotogramas para asignar una máscara a muchos.",
    },
    "  Imported mask for frame %1": {
        "zh_TW": "  已匯入影格 %1 的遮罩",
        "ja": "  フレーム %1 のマスクをインポートしました",
        "ko": "  프레임 %1의 마스크를 가져왔습니다",
        "de": "  Maske für Bild %1 importiert",
        "fr": "  Masque importé pour l'image %1",
        "es": "  Máscara importada para el fotograma %1",
    },
    "Batch import: %n mask(s) loaded": {
        "zh_TW": "批次匯入：已載入 %n 個遮罩",
        "ja": "バッチインポート: %n 個のマスクを読み込みました",
        "ko": "일괄 가져오기: 마스크 %n개를 불러왔습니다",
        "de": "Stapelimport: %n Maske(n) geladen",
        "fr": "Import par lot : %n masque(s) chargé(s)",
        "es": "Importación por lotes: %n máscara(s) cargada(s)",
    },
    "Preview": {
        "zh_TW": "預覽", "ja": "プレビュー", "ko": "미리보기",
        "de": "Vorschau", "fr": "Aperçu", "es": "Vista previa",
    },
    "(no image)": {
        "zh_TW": "（無影像）", "ja": "（画像なし）", "ko": "(이미지 없음)",
        "de": "(kein Bild)", "fr": "(aucune image)", "es": "(sin imagen)",
    },
    "Image only": {
        "zh_TW": "僅影像", "ja": "画像のみ", "ko": "이미지만",
        "de": "Nur Bild", "fr": "Image seule", "es": "Solo imagen",
    },
    "Image + Mask": {
        "zh_TW": "影像 + 遮罩", "ja": "画像 + マスク", "ko": "이미지 + 마스크",
        "de": "Bild + Maske", "fr": "Image + masque", "es": "Imagen + máscara",
    },
    "Mask only": {
        "zh_TW": "僅遮罩", "ja": "マスクのみ", "ko": "마스크만",
        "de": "Nur Maske", "fr": "Masque seul", "es": "Solo máscara",
    },
    "View:": {
        "zh_TW": "檢視：", "ja": "表示:", "ko": "보기:",
        "de": "Ansicht:", "fr": "Affichage :", "es": "Vista:",
    },
    "Alpha:": {
        "zh_TW": "透明度：", "ja": "アルファ:", "ko": "알파:",
        "de": "Alpha:", "fr": "Alpha :", "es": "Alfa:",
    },
    "Blue": {
        "zh_TW": "藍色", "ja": "青", "ko": "파랑",
        "de": "Blau", "fr": "Bleu", "es": "Azul",
    },
    "Red": {
        "zh_TW": "紅色", "ja": "赤", "ko": "빨강",
        "de": "Rot", "fr": "Rouge", "es": "Rojo",
    },
    "Green": {
        "zh_TW": "綠色", "ja": "緑", "ko": "초록",
        "de": "Grün", "fr": "Vert", "es": "Verde",
    },
    "Yellow": {
        "zh_TW": "黃色", "ja": "黄", "ko": "노랑",
        "de": "Gelb", "fr": "Jaune", "es": "Amarillo",
    },
    "Mask color:": {
        "zh_TW": "遮罩顏色：", "ja": "マスクの色:", "ko": "마스크 색상:",
        "de": "Maskenfarbe:", "fr": "Couleur du masque :", "es": "Color de máscara:",
    },
    "No mask assigned": {
        "zh_TW": "未指派遮罩", "ja": "マスク未割り当て", "ko": "지정된 마스크 없음",
        "de": "Keine Maske zugewiesen", "fr": "Aucun masque attribué",
        "es": "Sin máscara asignada",
    },
    "Frame %1 — %2": {
        "zh_TW": "影格 %1 — %2", "ja": "フレーム %1 — %2", "ko": "프레임 %1 — %2",
        "de": "Bild %1 — %2", "fr": "Image %1 — %2", "es": "Fotograma %1 — %2",
    },
    "Failed to load image": {
        "zh_TW": "無法載入影像", "ja": "画像の読み込みに失敗しました",
        "ko": "이미지를 불러오지 못했습니다",
        "de": "Bild konnte nicht geladen werden",
        "fr": "Échec du chargement de l'image",
        "es": "No se pudo cargar la imagen",
    },

    # ========== AdvancedTuningWidget ==========
    "AL-DIC Iterations": {
        "zh_CN": "AL-DIC 迭代次数",
        "zh_TW": "AL-DIC 迭代次數",
        "ja":    "AL-DIC 反復回数",
        "ko":    "AL-DIC 반복 횟수",
        "de":    "AL-DIC-Iterationen",
        "fr":    "Itérations AL-DIC",
        "es":    "Iteraciones AL-DIC",
    },
    "Number of global refinement cycles for the AL-DIC solver.\n"
    "1 = single global pass (fastest), 3 = default,\n"
    "5+ = diminishing returns for most cases.": {
        "zh_CN": "AL-DIC 求解器的全局精修迭代次数。\n"
                 "1 = 单次全局求解（最快），3 = 默认值，\n"
                 "5 次以上大多数情况下收益递减。",
        "zh_TW": "AL-DIC 求解器的全域精修迭代次數。\n"
                 "1 = 單次全域求解（最快），3 = 預設值，\n"
                 "5 次以上大多數情況下收益遞減。",
        "ja":    "AL-DIC ソルバーの全体的な精密化反復回数。\n"
                 "1 = 単一パス（最速）、3 = デフォルト、\n"
                 "5 以上はほとんどの場合で効果逓減。",
        "ko":    "AL-DIC 솔버의 전역 세분화 반복 횟수.\n"
                 "1 = 단일 패스(가장 빠름), 3 = 기본값,\n"
                 "5 이상은 대부분의 경우 수익이 감소합니다.",
        "de":    "Anzahl globaler Verfeinerungszyklen für den AL-DIC-Solver.\n"
                 "1 = einmaliger Durchlauf (schnellste), 3 = Standard,\n"
                 "5+ = abnehmender Ertrag in den meisten Fällen.",
        "fr":    "Nombre de cycles de raffinement global du solveur AL-DIC.\n"
                 "1 = passe unique (le plus rapide), 3 = par défaut,\n"
                 "5+ = rendement décroissant dans la plupart des cas.",
        "es":    "Número de ciclos de refinamiento global del solucionador AL-DIC.\n"
                 "1 = pasada única (más rápido), 3 = predeterminado,\n"
                 "5+ = rendimientos decrecientes en la mayoría de los casos.",
    },

    # ========== ExportDialog — column headers / short labels ==========
    "Auto": {
        "zh_CN": "自动", "zh_TW": "自動", "ja": "自動", "ko": "자동",
        "de": "Auto", "fr": "Auto", "es": "Auto",
    },
    "Opacity": {
        "zh_CN": "不透明度", "zh_TW": "不透明度", "ja": "不透明度",
        "ko": "불투명도", "de": "Deckkraft", "fr": "Opacité", "es": "Opacidad",
    },
    "Field opacity (0 = transparent, 1 = fully opaque)": {
        "zh_CN": "字段不透明度（0 = 透明，1 = 完全不透明）",
        "zh_TW": "欄位不透明度（0 = 透明，1 = 完全不透明）",
        "ja":    "フィールドの不透明度（0 = 透明、1 = 完全に不透明）",
        "ko":    "필드 불투명도 (0 = 투명, 1 = 완전 불투명)",
        "de":    "Feld-Deckkraft (0 = transparent, 1 = vollständig deckend)",
        "fr":    "Opacité du champ (0 = transparent, 1 = opaque)",
        "es":    "Opacidad del campo (0 = transparente, 1 = completamente opaco)",
    },
    "Edges": {
        "zh_CN": "边缘", "zh_TW": "邊緣", "ja": "縁",
        "ko": "가장자리", "de": "Ränder", "fr": "Bords", "es": "Bordes",
    },
    "Fill trimmed edges (display only)": {
        "zh_CN": "填充修剪的边缘（仅显示）",
        "zh_TW": "填充修剪的邊緣（僅顯示）",
        "ja":    "トリミングされた縁を補間（表示のみ）",
        "ko":    "잘라낸 가장자리 채우기 (표시 전용)",
        "de":    "Beschnittene Ränder füllen (nur Anzeige)",
        "fr":    "Remplir les bords rognés (affichage uniquement)",
        "es":    "Rellenar bordes recortados (solo visualización)",
    },
    "Re-interpolate the edge-trimmed strain band from reliable "
    "interior nodes. Affects the on-screen view and exported "
    "images/animations; exported data files always keep the "
    "trimmed edge as NaN.": {
        "zh_CN": "从可靠的内部节点重新插值被边缘修剪的应变带。"
                 "影响屏幕显示和导出的图片/动画；导出的数据文件"
                 "始终将修剪的边缘保留为 NaN。",
        "zh_TW": "從可靠的內部節點重新插值被邊緣修剪的應變帶。"
                 "影響螢幕顯示與匯出的圖片/動畫；匯出的資料檔案"
                 "始終將修剪的邊緣保留為 NaN。",
        "ja":    "縁がトリミングされたひずみ帯を、信頼できる内部ノードから"
                 "再補間します。画面表示とエクスポートした画像/アニメーション"
                 "に影響します。エクスポートしたデータファイルでは、"
                 "トリミングされた縁は常に NaN のままです。",
        "ko":    "가장자리가 잘린 변형률 띠를 신뢰할 수 있는 내부 노드에서 "
                 "다시 보간합니다. 화면 표시와 내보낸 이미지/애니메이션에 "
                 "영향을 줍니다. 내보낸 데이터 파일은 잘라낸 가장자리를 "
                 "항상 NaN으로 유지합니다.",
        "de":    "Interpoliert das randbeschnittene Dehnungsband aus "
                 "zuverlässigen inneren Knoten neu. Betrifft die "
                 "Bildschirmansicht und exportierte Bilder/Animationen; "
                 "exportierte Datendateien behalten den beschnittenen Rand "
                 "immer als NaN.",
        "fr":    "Réinterpole la bande de déformation rognée aux bords à "
                 "partir de nœuds intérieurs fiables. Affecte l'affichage à "
                 "l'écran et les images/animations exportées ; les fichiers "
                 "de données exportés conservent toujours le bord rogné en NaN.",
        "es":    "Reinterpola la banda de deformación recortada en los bordes "
                 "a partir de nodos interiores fiables. Afecta a la vista en "
                 "pantalla y a las imágenes/animaciones exportadas; los "
                 "archivos de datos exportados siempre mantienen el borde "
                 "recortado como NaN.",
    },
    "Export": {
        "zh_CN": "导出", "zh_TW": "匯出", "ja": "エクスポート",
        "ko": "내보내기", "de": "Exportieren", "fr": "Exporter", "es": "Exportar",
    },
    "Field": {
        "zh_CN": "字段", "zh_TW": "欄位", "ja": "フィールド",
        "ko": "필드", "de": "Feld", "fr": "Champ", "es": "Campo",
    },
    "Colormap": {
        "zh_CN": "颜色映射", "zh_TW": "色彩對映", "ja": "カラーマップ",
        "ko": "색상 맵", "de": "Farbskala", "fr": "Palette", "es": "Mapa de colores",
    },
    "Min": {
        "zh_CN": "最小", "zh_TW": "最小", "ja": "最小",
        "ko": "최소", "de": "Min", "fr": "Min", "es": "Mín",
    },
    "Max": {
        "zh_CN": "最大", "zh_TW": "最大", "ja": "最大",
        "ko": "최대", "de": "Max", "fr": "Max", "es": "Máx",
    },
    "IMAGE SETTINGS": {
        "zh_CN": "图像设置", "zh_TW": "影像設定", "ja": "画像設定",
        "ko": "이미지 설정", "de": "BILDEINSTELLUNGEN",
        "fr": "PARAMÈTRES D'IMAGE", "es": "AJUSTES DE IMAGEN",
    },
    "Format": {
        "zh_CN": "格式", "zh_TW": "格式", "ja": "形式",
        "ko": "형식", "de": "Format", "fr": "Format", "es": "Formato",
    },
    "DPI": {
        "zh_CN": "DPI", "zh_TW": "DPI", "ja": "DPI",
        "ko": "DPI", "de": "DPI", "fr": "DPI", "es": "PPP",
    },
    # ========== Export: output resolution + JPEG quality (all 8 locales) ==========
    "Resolution (long edge)": {
        "zh_CN": "分辨率（长边）", "zh_TW": "解析度（長邊）",
        "ja": "解像度（長辺）", "ko": "해상도(긴 변)",
        "de": "Auflösung (lange Kante)", "fr": "Résolution (bord long)",
        "es": "Resolución (borde largo)",
    },
    "Frame step": {
        "zh_CN": "抽帧间隔", "zh_TW": "抽幀間隔", "ja": "フレーム間引き",
        "ko": "프레임 간격", "de": "Bildschritt", "fr": "Pas d'image",
        "es": "Paso de fotogramas",
    },
    "Full resolution": {
        "zh_CN": "原始分辨率", "zh_TW": "原始解析度", "ja": "フル解像度",
        "ko": "전체 해상도", "de": "Volle Auflösung",
        "fr": "Résolution native", "es": "Resolución completa",
    },
    "JPEG quality": {
        "zh_CN": "JPEG 质量", "zh_TW": "JPEG 品質", "ja": "JPEG 品質",
        "ko": "JPEG 품질", "de": "JPEG-Qualität", "fr": "Qualité JPEG",
        "es": "Calidad JPEG",
    },
    "JPEG quality (higher = larger file). Ignored for PNG/TIFF.": {
        "zh_CN": "JPEG 质量（越高文件越大）。对 PNG/TIFF 无效。",
        "zh_TW": "JPEG 品質（越高檔案越大）。對 PNG/TIFF 無效。",
        "ja": "JPEG 品質（高いほどファイルが大きくなります）。PNG/TIFF では無視されます。",
        "ko": "JPEG 품질(높을수록 파일이 커집니다). PNG/TIFF에서는 무시됩니다.",
        "de": "JPEG-Qualität (höher = größere Datei). Wird für PNG/TIFF ignoriert.",
        "fr": "Qualité JPEG (plus élevée = fichier plus gros). Ignorée pour PNG/TIFF.",
        "es": "Calidad JPEG (mayor = archivo más grande). Se ignora para PNG/TIFF.",
    },
    "Cap the exported image's long edge (the larger of width/height; "
    "aspect ratio is kept).\nField detail is bounded by the mesh, so a "
    "smaller cap is near-lossless\nbut much smaller on disk and faster "
    "to encode. Lower = faster. 'Full resolution' keeps the native "
    "size.": {
        "zh_CN": "限制导出图像的长边（宽和高中较大的一个；保持宽高比）。\n"
                 "场的细节由网格密度决定，因此较小的上限几乎无损，\n"
                 "但文件更小、编码更快。越低越快。「原始分辨率」保持原生尺寸。",
        "zh_TW": "限制匯出影像的長邊（寬與高中較大的一個；保持長寬比）。\n"
                 "場的細節由網格密度決定，因此較小的上限幾乎無損，\n"
                 "但檔案更小、編碼更快。越低越快。「原始解析度」保持原生尺寸。",
        "ja": "書き出す画像の長辺（幅と高さの大きい方。縦横比は維持）を制限します。\n"
              "フィールドの詳細はメッシュで決まるため、上限を小さくしてもほぼ無損失で、\n"
              "ファイルは小さく書き出しも高速です。小さいほど高速。「フル解像度」は元のサイズを保ちます。",
        "ko": "내보내는 이미지의 긴 변(너비/높이 중 큰 값, 종횡비 유지)을 제한합니다.\n"
              "필드 세부 정보는 메시로 결정되므로 상한을 낮춰도 거의 무손실이며,\n"
              "파일이 작고 인코딩이 빠릅니다. 낮을수록 빠름. '전체 해상도'는 원본 크기를 유지합니다.",
        "de": "Begrenzt die lange Kante des exportierten Bildes (das Größere von Breite/Höhe; Seitenverhältnis bleibt erhalten).\n"
              "Die Felddetails sind durch das Netz begrenzt, daher ist eine kleinere Grenze nahezu verlustfrei,\n"
              "aber viel kleiner und schneller zu kodieren. Kleiner = schneller. „Volle Auflösung“ behält die native Größe bei.",
        "fr": "Limite le bord long de l'image exportée (le plus grand de largeur/hauteur ; le ratio est conservé).\n"
              "Le détail du champ est borné par le maillage, donc une limite plus petite est quasi sans perte,\n"
              "mais bien plus légère et rapide à encoder. Plus petit = plus rapide. « Résolution native » conserve la taille native.",
        "es": "Limita el borde largo de la imagen exportada (el mayor de ancho/alto; se mantiene la relación de aspecto).\n"
              "El detalle del campo está limitado por la malla, por lo que un límite menor es casi sin pérdida,\n"
              "pero mucho más pequeño y rápido de codificar. Menor = más rápido. «Resolución completa» mantiene el tamaño nativo.",
    },
    "Cap the animation's long edge (the larger of width/height).\n"
    "Lower = faster and much smaller. Strongly recommended for GIF, "
    "whose size explodes at native resolution.": {
        "zh_CN": "限制动画的长边（宽和高中较大的一个）。\n"
                 "越低越快、越小。强烈建议用于 GIF：其体积在原生分辨率下会急剧膨胀。",
        "zh_TW": "限制動畫的長邊（寬與高中較大的一個）。\n"
                 "越低越快、越小。強烈建議用於 GIF：其體積在原生解析度下會急劇膨脹。",
        "ja": "アニメーションの長辺（幅と高さの大きい方）を制限します。\n"
              "小さいほど高速・小容量。GIF に強く推奨されます。ネイティブ解像度ではサイズが急激に増大します。",
        "ko": "애니메이션의 긴 변(너비/높이 중 큰 값)을 제한합니다.\n"
              "낮을수록 빠르고 작습니다. GIF에 강력히 권장됩니다. 원본 해상도에서는 크기가 급격히 커집니다.",
        "de": "Begrenzt die lange Kante der Animation (das Größere von Breite/Höhe).\n"
              "Kleiner = schneller und viel kleiner. Dringend empfohlen für GIF, dessen Größe bei nativer Auflösung explodiert.",
        "fr": "Limite le bord long de l'animation (le plus grand de largeur/hauteur).\n"
              "Plus petit = plus rapide et bien plus léger. Fortement recommandé pour le GIF, dont la taille explose en résolution native.",
        "es": "Limita el borde largo de la animación (el mayor de ancho/alto).\n"
              "Menor = más rápido y mucho más pequeño. Muy recomendable para GIF, cuyo tamaño se dispara a resolución nativa.",
    },
    "Export every Nth frame (1 = every frame). Higher is faster and "
    "smaller\nbut looks choppier. Playback duration is preserved (the "
    "FPS above is the pre-decimation rate).": {
        "zh_CN": "每 N 帧导出一帧（1 = 每帧都导出）。越大越快、越小，\n"
                 "但看起来更卡顿。播放时长保持不变（上方 FPS 为抽帧前的帧率）。",
        "zh_TW": "每 N 幀匯出一幀（1 = 每幀都匯出）。越大越快、越小，\n"
                 "但看起來更卡頓。播放時長保持不變（上方 FPS 為抽幀前的幀率）。",
        "ja": "N フレームごとに 1 枚書き出します（1 = 全フレーム）。大きいほど高速・小容量ですが、\n"
              "カクついて見えます。再生時間は維持されます（上の FPS は間引き前のレート）。",
        "ko": "N 프레임마다 하나씩 내보냅니다(1 = 모든 프레임). 클수록 빠르고 작지만,\n"
              "더 끊겨 보입니다. 재생 시간은 유지됩니다(위의 FPS는 추출 전 프레임률).",
        "de": "Jedes N-te Bild exportieren (1 = jedes Bild). Höher = schneller und kleiner,\n"
              "wirkt aber ruckeliger. Die Abspieldauer bleibt erhalten (die FPS oben sind die Rate vor der Dezimierung).",
        "fr": "Exporte une image sur N (1 = toutes les images). Plus élevé = plus rapide et plus léger,\n"
              "mais plus saccadé. La durée de lecture est conservée (les FPS ci-dessus sont le débit avant décimation).",
        "es": "Exporta uno de cada N fotogramas (1 = todos). Mayor = más rápido y pequeño,\n"
              "pero se ve más entrecortado. La duración se conserva (los FPS de arriba son la tasa antes de diezmar).",
    },
    # ========== Export: Preview & Colorbar tab (all 8 locales) ==========
    "Preview & Colorbar": {
        "zh_CN": "预览与色条", "zh_TW": "預覽與色條",
        "ja": "プレビューとカラーバー", "ko": "미리보기 및 컬러바",
        "de": "Vorschau & Farbleiste", "fr": "Aperçu et barre de couleur",
        "es": "Vista previa y barra de color",
    },
    "COLORBAR STYLE": {
        "zh_CN": "色条样式", "zh_TW": "色條樣式", "ja": "カラーバーのスタイル",
        "ko": "컬러바 스타일", "de": "FARBLEISTEN-STIL",
        "fr": "STYLE DE BARRE DE COULEUR", "es": "ESTILO DE BARRA DE COLOR",
    },
    "Frame": {
        "zh_CN": "帧", "zh_TW": "影格", "ja": "フレーム", "ko": "프레임",
        "de": "Bild", "fr": "Image", "es": "Fotograma",
    },
    "Position": {
        "zh_CN": "位置", "zh_TW": "位置", "ja": "位置", "ko": "위치",
        "de": "Position", "fr": "Position", "es": "Posición",
    },
    "Right": {
        "zh_CN": "右", "zh_TW": "右", "ja": "右", "ko": "오른쪽",
        "de": "Rechts", "fr": "Droite", "es": "Derecha",
    },
    "Left": {
        "zh_CN": "左", "zh_TW": "左", "ja": "左", "ko": "왼쪽",
        "de": "Links", "fr": "Gauche", "es": "Izquierda",
    },
    "Top": {
        "zh_CN": "上", "zh_TW": "上", "ja": "上", "ko": "위",
        "de": "Oben", "fr": "Haut", "es": "Arriba",
    },
    "Bottom": {
        "zh_CN": "下", "zh_TW": "下", "ja": "下", "ko": "아래",
        "de": "Unten", "fr": "Bas", "es": "Abajo",
    },
    "Font size": {
        "zh_CN": "字号", "zh_TW": "字級", "ja": "フォントサイズ",
        "ko": "글꼴 크기", "de": "Schriftgröße", "fr": "Taille de police",
        "es": "Tamaño de fuente",
    },
    "Font family": {
        "zh_CN": "字体", "zh_TW": "字型", "ja": "フォント",
        "ko": "글꼴", "de": "Schriftart", "fr": "Police",
        "es": "Fuente",
    },
    "Bar thickness": {
        "zh_CN": "色条粗细", "zh_TW": "色條粗細", "ja": "バーの太さ",
        "ko": "막대 두께", "de": "Balkendicke", "fr": "Épaisseur de la barre",
        "es": "Grosor de la barra",
    },
    "Background": {
        "zh_CN": "背景", "zh_TW": "背景", "ja": "背景", "ko": "배경",
        "de": "Hintergrund", "fr": "Arrière-plan", "es": "Fondo",
    },
    "Black": {
        "zh_CN": "黑色", "zh_TW": "黑色", "ja": "黒", "ko": "검정",
        "de": "Schwarz", "fr": "Noir", "es": "Negro",
    },
    "White": {
        "zh_CN": "白色", "zh_TW": "白色", "ja": "白", "ko": "흰색",
        "de": "Weiß", "fr": "Blanc", "es": "Blanco",
    },
    "Refresh preview": {
        "zh_CN": "刷新预览", "zh_TW": "重新整理預覽", "ja": "プレビューを更新",
        "ko": "미리보기 새로고침", "de": "Vorschau aktualisieren",
        "fr": "Actualiser l'aperçu", "es": "Actualizar vista previa",
    },
    "FIELD APPEARANCE": {
        "zh_CN": "字段外观", "zh_TW": "欄位外觀", "ja": "フィールドの外観",
        "ko": "필드 모양", "de": "FELDDARSTELLUNG",
        "fr": "APPARENCE DU CHAMP", "es": "APARIENCIA DEL CAMPO",
    },
    "Open this tab to render a preview.": {
        "zh_CN": "打开此选项卡以渲染预览。",
        "zh_TW": "開啟此分頁以算繪預覽。",
        "ja": "このタブを開くとプレビューが描画されます。",
        "ko": "이 탭을 열면 미리보기가 렌더링됩니다.",
        "de": "Diesen Reiter öffnen, um eine Vorschau zu rendern.",
        "fr": "Ouvrez cet onglet pour générer un aperçu.",
        "es": "Abre esta pestaña para generar una vista previa.",
    },
    "Enable a field on the Images tab to preview.": {
        "zh_CN": "在 Images 页启用一个字段以进行预览。",
        "zh_TW": "在 Images 頁啟用一個欄位以進行預覽。",
        "ja": "プレビューするには Images タブでフィールドを有効にしてください。",
        "ko": "미리보려면 Images 탭에서 필드를 활성화하세요.",
        "de": "Aktivieren Sie ein Feld im Reiter „Images“ für die Vorschau.",
        "fr": "Activez un champ dans l'onglet Images pour l'aperçu.",
        "es": "Active un campo en la pestaña Images para la vista previa.",
    },
    "No data for this field/frame.": {
        "zh_CN": "该字段/帧没有数据。",
        "zh_TW": "該欄位/影格沒有資料。",
        "ja": "このフィールド/フレームにはデータがありません。",
        "ko": "이 필드/프레임에 데이터가 없습니다.",
        "de": "Keine Daten für dieses Feld/Bild.",
        "fr": "Aucune donnée pour ce champ/cette image.",
        "es": "No hay datos para este campo/fotograma.",
    },
    "Preview failed: ": {
        "zh_CN": "预览失败：", "zh_TW": "預覽失敗：",
        "ja": "プレビューに失敗しました：", "ko": "미리보기 실패: ",
        "de": "Vorschau fehlgeschlagen: ", "fr": "Échec de l'aperçu : ",
        "es": "Error en la vista previa: ",
    },
    # ========== Export: apply-to-all + margin (all 8 locales) ==========
    "Apply to all fields": {
        "zh_CN": "应用到所有字段", "zh_TW": "套用到所有欄位",
        "ja": "すべてのフィールドに適用", "ko": "모든 필드에 적용",
        "de": "Auf alle Felder anwenden", "fr": "Appliquer à tous les champs",
        "es": "Aplicar a todos los campos",
    },
    "Apply this field's colormap, opacity and auto-range to every "
    "enabled field (each field keeps its own min/max).": {
        "zh_CN": "将该字段的 colormap、不透明度和自动范围应用到所有已启用字段（每个字段保留各自的 min/max）。",
        "zh_TW": "將該欄位的 colormap、不透明度和自動範圍套用到所有已啟用欄位（每個欄位保留各自的 min/max）。",
        "ja": "このフィールドの colormap・不透明度・自動範囲を、有効なすべてのフィールドに適用します（各フィールドの min/max は保持）。",
        "ko": "이 필드의 colormap, 불투명도, 자동 범위를 활성화된 모든 필드에 적용합니다(각 필드의 min/max는 유지).",
        "de": "Colormap, Deckkraft und Auto-Bereich dieses Felds auf alle aktivierten Felder anwenden (jedes Feld behält sein eigenes Min/Max).",
        "fr": "Applique la colormap, l'opacité et l'auto-plage de ce champ à tous les champs activés (chaque champ garde ses propres min/max).",
        "es": "Aplica el colormap, la opacidad y el rango automático de este campo a todos los campos activados (cada campo conserva su propio mín/máx).",
    },
    "Margin": {
        "zh_CN": "边距", "zh_TW": "邊距", "ja": "余白", "ko": "여백",
        "de": "Rand", "fr": "Marge", "es": "Margen",
    },
    "Margin color": {
        "zh_CN": "边距颜色", "zh_TW": "邊距顏色", "ja": "余白の色",
        "ko": "여백 색상", "de": "Randfarbe", "fr": "Couleur de marge",
        "es": "Color del margen",
    },
    "Add a blank border around the exported content, as a fraction of "
    "the long edge (0 = none).": {
        "zh_CN": "在导出内容外围加一圈空白边框，宽度为长边的比例（0 = 无）。",
        "zh_TW": "在匯出內容外圍加一圈空白邊框，寬度為長邊的比例（0 = 無）。",
        "ja": "書き出す内容の周囲に空白の枠を追加します。幅は長辺に対する割合です（0 = なし）。",
        "ko": "내보내는 콘텐츠 주위에 여백 테두리를 추가합니다. 너비는 긴 변에 대한 비율입니다(0 = 없음).",
        "de": "Fügt einen leeren Rand um den exportierten Inhalt hinzu, als Anteil der langen Kante (0 = keiner).",
        "fr": "Ajoute une bordure vide autour du contenu exporté, en fraction du bord long (0 = aucune).",
        "es": "Añade un borde en blanco alrededor del contenido exportado, como fracción del borde largo (0 = ninguna).",
    },
    # ========== Session persistence (results + file association) ==========
    "Associate .aldic files with pyALDIC…": {
        "zh_CN": "将 .aldic 文件关联到 pyALDIC…",
        "zh_TW": "將 .aldic 檔案關聯到 pyALDIC…",
        "ja": ".aldic ファイルを pyALDIC に関連付け…",
        "ko": ".aldic 파일을 pyALDIC에 연결…",
        "de": ".aldic-Dateien mit pyALDIC verknüpfen…",
        "fr": "Associer les fichiers .aldic à pyALDIC…",
        "es": "Asociar archivos .aldic con pyALDIC…",
    },
    "Register .aldic so double-clicking a session file opens pyALDIC "
    "(current user only, no admin rights needed).": {
        "zh_CN": "注册 .aldic，让双击会话文件即可打开 pyALDIC（仅当前用户，无需管理员权限）。",
        "zh_TW": "註冊 .aldic，讓雙擊工作階段檔案即可開啟 pyALDIC（僅目前使用者，無需系統管理員權限）。",
        "ja": ".aldic を登録し、セッションファイルをダブルクリックすると pyALDIC が開くようにします（現在のユーザーのみ、管理者権限は不要）。",
        "ko": ".aldic를 등록하여 세션 파일을 두 번 클릭하면 pyALDIC가 열리도록 합니다(현재 사용자만, 관리자 권한 불필요).",
        "de": "Registriert .aldic, sodass ein Doppelklick auf eine Sitzungsdatei pyALDIC öffnet (nur aktueller Benutzer, keine Administratorrechte nötig).",
        "fr": "Enregistre .aldic pour qu'un double-clic sur un fichier de session ouvre pyALDIC (utilisateur actuel uniquement, sans droits administrateur).",
        "es": "Registra .aldic para que hacer doble clic en un archivo de sesión abra pyALDIC (solo el usuario actual, sin permisos de administrador).",
    },
    "Include Results?": {
        "zh_CN": "包含结果？", "zh_TW": "包含結果？", "ja": "結果を含めますか？",
        "ko": "결과를 포함할까요?", "de": "Ergebnisse einbeziehen?",
        "fr": "Inclure les résultats ?", "es": "¿Incluir resultados?",
    },
    "Include the computed results in this session?": {
        "zh_CN": "在此会话中包含已计算的结果吗？",
        "zh_TW": "在此工作階段中包含已計算的結果嗎？",
        "ja": "このセッションに計算済みの結果を含めますか？",
        "ko": "이 세션에 계산된 결과를 포함하시겠습니까?",
        "de": "Die berechneten Ergebnisse in diese Sitzung einbeziehen?",
        "fr": "Inclure les résultats calculés dans cette session ?",
        "es": "¿Incluir los resultados calculados en esta sesión?",
    },
    "Including results (about %1 uncompressed) lets you reopen the "
    "session without recomputing. Choose No to save a small "
    "configuration-only file for sharing.": {
        "zh_CN": "包含结果（未压缩约 %1）可让你下次直接打开会话而无需重新计算。选择“否”则只保存一个小的仅配置文件，便于分享。",
        "zh_TW": "包含結果（未壓縮約 %1）可讓你下次直接開啟工作階段而無需重新計算。選擇「否」則只儲存一個小的僅設定檔案，便於分享。",
        "ja": "結果を含めると（非圧縮で約 %1）、再計算せずにセッションを再度開けます。「いいえ」を選ぶと、共有用に設定のみの小さなファイルを保存します。",
        "ko": "결과를 포함하면(압축 전 약 %1) 다시 계산하지 않고 세션을 다시 열 수 있습니다. '아니요'를 선택하면 공유용으로 구성만 담긴 작은 파일을 저장합니다.",
        "de": "Mit Ergebnissen (etwa %1 unkomprimiert) können Sie die Sitzung ohne Neuberechnung wieder öffnen. Wählen Sie Nein, um eine kleine reine Konfigurationsdatei zum Teilen zu speichern.",
        "fr": "Inclure les résultats (environ %1 non compressé) permet de rouvrir la session sans tout recalculer. Choisissez Non pour enregistrer un petit fichier de configuration seule, à partager.",
        "es": "Incluir los resultados (unos %1 sin comprimir) permite reabrir la sesión sin recalcular. Elija No para guardar un pequeño archivo solo de configuración para compartir.",
    },
    "large": {
        "zh_CN": "较大", "zh_TW": "較大", "ja": "大きい", "ko": "큼",
        "de": "groß", "fr": "volumineux", "es": "grande",
    },
    "Saving Session": {
        "zh_CN": "正在保存会话", "zh_TW": "正在儲存工作階段",
        "ja": "セッションを保存中", "ko": "세션 저장 중",
        "de": "Sitzung wird gespeichert", "fr": "Enregistrement de la session",
        "es": "Guardando sesión",
    },
    "Loading Session": {
        "zh_CN": "正在加载会话", "zh_TW": "正在載入工作階段",
        "ja": "セッションを読み込み中", "ko": "세션 불러오는 중",
        "de": "Sitzung wird geladen", "fr": "Chargement de la session",
        "es": "Cargando sesión",
    },
    "File Association Failed": {
        "zh_CN": "文件关联失败", "zh_TW": "檔案關聯失敗",
        "ja": "ファイル関連付けに失敗", "ko": "파일 연결 실패",
        "de": "Dateiverknüpfung fehlgeschlagen",
        "fr": "Échec de l'association de fichiers",
        "es": "Error al asociar archivos",
    },
    "Could not register .aldic files: ": {
        "zh_CN": "无法注册 .aldic 文件：", "zh_TW": "無法註冊 .aldic 檔案：",
        "ja": ".aldic ファイルを登録できませんでした：",
        "ko": ".aldic 파일을 등록할 수 없습니다: ",
        "de": ".aldic-Dateien konnten nicht registriert werden: ",
        "fr": "Impossible d'enregistrer les fichiers .aldic : ",
        "es": "No se pudieron registrar los archivos .aldic: ",
    },
    "File Association": {
        "zh_CN": "文件关联", "zh_TW": "檔案關聯", "ja": "ファイル関連付け",
        "ko": "파일 연결", "de": "Dateiverknüpfung",
        "fr": "Association de fichiers", "es": "Asociación de archivos",
    },
    "Done. Double-clicking a .aldic file will now open pyALDIC and "
    "restore that session.": {
        "zh_CN": "完成。现在双击 .aldic 文件即可打开 pyALDIC 并恢复该会话。",
        "zh_TW": "完成。現在雙擊 .aldic 檔案即可開啟 pyALDIC 並還原該工作階段。",
        "ja": "完了しました。これで .aldic ファイルをダブルクリックすると pyALDIC が開き、そのセッションが復元されます。",
        "ko": "완료되었습니다. 이제 .aldic 파일을 두 번 클릭하면 pyALDIC가 열리고 해당 세션이 복원됩니다.",
        "de": "Fertig. Ein Doppelklick auf eine .aldic-Datei öffnet nun pyALDIC und stellt diese Sitzung wieder her.",
        "fr": "Terminé. Un double-clic sur un fichier .aldic ouvrira désormais pyALDIC et restaurera cette session.",
        "es": "Listo. Ahora, hacer doble clic en un archivo .aldic abrirá pyALDIC y restaurará esa sesión.",
    },
    "Include colorbar": {
        "zh_CN": "包含色条", "zh_TW": "包含色條", "ja": "カラーバーを含める",
        "ko": "컬러바 포함", "de": "Farbleiste einfügen",
        "fr": "Inclure la barre de couleur", "es": "Incluir barra de color",
    },
    "Append a vertical colorbar strip to the right of each image.\n"
    "Tick labels update per frame when Auto range is enabled.": {
        "zh_CN": "在每张图像右侧添加一条垂直色条。\n"
                 "启用自动范围时，刻度标签会按帧更新。",
        "zh_TW": "在每張影像右側添加一條垂直色條。\n"
                 "啟用自動範圍時，刻度標籤會依影格更新。",
        "ja":    "各画像の右側に垂直カラーバーを追加します。\n"
                 "自動レンジ有効時、目盛りラベルはフレームごとに更新されます。",
        "ko":    "각 이미지 오른쪽에 수직 컬러바를 추가합니다.\n"
                 "자동 범위가 활성화되면 눈금 레이블이 프레임별로 갱신됩니다.",
        "de":    "Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.\n"
                 "Die Beschriftungen aktualisieren sich pro Bild, wenn Auto aktiv ist.",
        "fr":    "Ajoute une barre de couleur verticale à droite de chaque image.\n"
                 "Les étiquettes se mettent à jour par image quand la plage auto est activée.",
        "es":    "Añade una barra de color vertical a la derecha de cada imagen.\n"
                 "Las etiquetas se actualizan por fotograma cuando el rango auto está activo.",
    },
    "Append a vertical colorbar strip to the right of each frame.\n"
    "Tick labels update per frame when Auto range is enabled.": {
        "zh_CN": "在每一帧右侧添加一条垂直色条。\n"
                 "启用自动范围时，刻度标签会按帧更新。",
        "zh_TW": "在每一影格右側添加一條垂直色條。\n"
                 "啟用自動範圍時，刻度標籤會依影格更新。",
        "ja":    "各フレームの右側に垂直カラーバーを追加します。\n"
                 "自動レンジ有効時、目盛りラベルはフレームごとに更新されます。",
        "ko":    "각 프레임 오른쪽에 수직 컬러바를 추가합니다.\n"
                 "자동 범위가 활성화되면 눈금 레이블이 프레임별로 갱신됩니다.",
        "de":    "Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.\n"
                 "Die Beschriftungen aktualisieren sich pro Bild, wenn Auto aktiv ist.",
        "fr":    "Ajoute une barre de couleur verticale à droite de chaque image.\n"
                 "Les étiquettes se mettent à jour par image quand la plage auto est activée.",
        "es":    "Añade una barra de color vertical a la derecha de cada fotograma.\n"
                 "Las etiquetas se actualizan por fotograma cuando el rango auto está activo.",
    },
    "Original (frame 1 background)": {
        "zh_CN": "原始配置（第 1 帧作背景）",
        "zh_TW": "原始配置（第 1 影格作背景）",
        "ja":    "原形（第 1 フレームを背景）",
        "ko":    "원형 (1번 프레임을 배경으로)",
        "de":    "Original (Bild 1 als Hintergrund)",
        "fr":    "Original (image 1 en arrière-plan)",
        "es":    "Original (fotograma 1 como fondo)",
    },
    "Field is drawn at the original (undeformed) node positions.\n"
    "Background image is always the first frame.": {
        "zh_CN": "字段绘制在原始（未变形）节点位置。\n"
                 "背景图像始终是第一帧。",
        "zh_TW": "欄位繪製在原始（未變形）節點位置。\n"
                 "背景影像始終是第一影格。",
        "ja":    "フィールドは元の（未変形の）ノード位置に描画されます。\n"
                 "背景画像は常に最初のフレームです。",
        "ko":    "필드는 원래(변형되지 않은) 노드 위치에 그려집니다.\n"
                 "배경 이미지는 항상 첫 프레임입니다.",
        "de":    "Feld wird an den ursprünglichen (unverformten) Knotenpositionen gezeichnet.\n"
                 "Das Hintergrundbild ist immer das erste Bild.",
        "fr":    "Le champ est tracé aux positions de nœud originales (non déformées).\n"
                 "L'image de fond est toujours la première image.",
        "es":    "El campo se dibuja en las posiciones originales (no deformadas) de los nodos.\n"
                 "La imagen de fondo es siempre el primer fotograma.",
    },
    "Deformed (current frame background)": {
        "zh_CN": "变形配置（当前帧作背景）",
        "zh_TW": "變形配置（當前影格作背景）",
        "ja":    "変形後（現在のフレームを背景）",
        "ko":    "변형 후 (현재 프레임을 배경으로)",
        "de":    "Verformt (aktuelles Bild als Hintergrund)",
        "fr":    "Déformé (image actuelle en arrière-plan)",
        "es":    "Deformado (fotograma actual como fondo)",
    },
    "Field is drawn at the displaced node positions "
    "(reference + displacement).\n"
    "Background image follows each frame's own photo.": {
        "zh_CN": "字段绘制在位移后节点位置（参考位置 + 位移）。\n"
                 "背景图像跟随每帧自身的照片。",
        "zh_TW": "欄位繪製在位移後節點位置（參考位置 + 位移）。\n"
                 "背景影像跟隨每影格自身的照片。",
        "ja":    "フィールドは変位後のノード位置（参照 + 変位）に描画されます。\n"
                 "背景画像は各フレーム自身の写真を使用します。",
        "ko":    "필드는 변위된 노드 위치(참조 + 변위)에 그려집니다.\n"
                 "배경 이미지는 각 프레임 자체의 사진을 따릅니다.",
        "de":    "Feld wird an den verschobenen Knotenpositionen (Referenz + Verschiebung) gezeichnet.\n"
                 "Das Hintergrundbild folgt dem Foto jedes Bildes.",
        "fr":    "Le champ est tracé aux positions de nœud déplacées (référence + déplacement).\n"
                 "L'image de fond suit la photo de chaque image.",
        "es":    "El campo se dibuja en las posiciones de nodo desplazadas (referencia + desplazamiento).\n"
                 "La imagen de fondo sigue la foto de cada fotograma.",
    },
    "Render as": {
        "zh_CN": "绘制为", "zh_TW": "繪製為", "ja": "描画方法",
        "ko": "렌더링 방식", "de": "Darstellen als",
        "fr": "Rendu", "es": "Representar como",
    },
    "Cancel Export": {
        "zh_CN": "取消导出", "zh_TW": "取消匯出",
        "ja": "エクスポートをキャンセル", "ko": "내보내기 취소",
        "de": "Export abbrechen", "fr": "Annuler l'export",
        "es": "Cancelar exportación",
    },
    "Export Images": {
        "zh_CN": "导出图像", "zh_TW": "匯出影像",
        "ja": "画像をエクスポート", "ko": "이미지 내보내기",
        "de": "Bilder exportieren", "fr": "Exporter les images",
        "es": "Exportar imágenes",
    },
    "ANIMATION SETTINGS": {
        "zh_CN": "动画设置", "zh_TW": "動畫設定",
        "ja": "アニメーション設定", "ko": "애니메이션 설정",
        "de": "ANIMATIONSEINSTELLUNGEN",
        "fr": "PARAMÈTRES D'ANIMATION", "es": "AJUSTES DE ANIMACIÓN",
    },
    "FPS": {
        "zh_CN": "帧率", "zh_TW": "影格率", "ja": "FPS",
        "ko": "FPS", "de": "FPS", "fr": "FPS", "es": "FPS",
    },
    "Export Animation": {
        "zh_CN": "导出动画", "zh_TW": "匯出動畫",
        "ja": "アニメーションをエクスポート", "ko": "애니메이션 내보내기",
        "de": "Animation exportieren", "fr": "Exporter l'animation",
        "es": "Exportar animación",
    },
    "CONTENT": {
        "zh_CN": "内容", "zh_TW": "內容", "ja": "内容",
        "ko": "내용", "de": "INHALT", "fr": "CONTENU", "es": "CONTENIDO",
    },
    "Parameter summary table": {
        "zh_CN": "参数摘要表", "zh_TW": "參數摘要表",
        "ja": "パラメータ要約表", "ko": "매개변수 요약 표",
        "de": "Parameter-Übersichtstabelle",
        "fr": "Tableau récapitulatif des paramètres",
        "es": "Tabla resumen de parámetros",
    },
    "Field statistics (min/max/mean/std per frame)": {
        "zh_CN": "字段统计（每帧 最小/最大/平均/标准差）",
        "zh_TW": "欄位統計（每影格 最小/最大/平均/標準差）",
        "ja":    "フィールド統計（フレームごとの最小/最大/平均/標準偏差）",
        "ko":    "필드 통계 (프레임별 최소/최대/평균/표준편차)",
        "de":    "Feldstatistik (min/max/Mittelwert/Stdabw. pro Bild)",
        "fr":    "Statistiques de champ (min/max/moyenne/écart-type par image)",
        "es":    "Estadísticas de campo (mín/máx/media/desv.típ. por fotograma)",
    },
    "Sample field images": {
        "zh_CN": "示例字段图像", "zh_TW": "範例欄位影像",
        "ja": "フィールド画像のサンプル", "ko": "필드 이미지 샘플",
        "de": "Beispiel-Feldbilder", "fr": "Exemples d'images de champ",
        "es": "Imágenes de campo de muestra",
    },
    "Sample every": {
        "zh_CN": "每隔", "zh_TW": "每隔",
        "ja": "抽出間隔", "ko": "샘플 간격",
        "de": "Alle", "fr": "Échantillonner toutes les", "es": "Muestrear cada",
    },
    "frames": {
        "zh_CN": "帧", "zh_TW": "影格",
        "ja": "フレーム", "ko": "프레임",
        "de": "Bilder", "fr": "images", "es": "fotogramas",
    },
    "FIELDS": {
        "zh_CN": "字段", "zh_TW": "欄位", "ja": "フィールド",
        "ko": "필드", "de": "FELDER", "fr": "CHAMPS", "es": "CAMPOS",
    },
    "Displacement:": {
        "zh_CN": "位移：", "zh_TW": "位移：",
        "ja": "変位：", "ko": "변위:",
        "de": "Verschiebung:", "fr": "Déplacement :", "es": "Desplazamiento:",
    },
    "Strain:": {
        "zh_CN": "应变：", "zh_TW": "應變：",
        "ja": "ひずみ：", "ko": "변형률:",
        "de": "Dehnung:", "fr": "Déformation :", "es": "Deformación:",
    },
    "Format: HTML (self-contained, view in any browser)": {
        "zh_CN": "格式：HTML（自包含，可在任意浏览器中查看）",
        "zh_TW": "格式：HTML（自包含，可在任意瀏覽器中檢視）",
        "ja":    "形式：HTML（自己完結型、任意のブラウザで表示可能）",
        "ko":    "형식: HTML (자체 포함, 모든 브라우저에서 볼 수 있음)",
        "de":    "Format: HTML (eigenständig, in jedem Browser anzeigbar)",
        "fr":    "Format : HTML (autonome, consultable dans n'importe quel navigateur)",
        "es":    "Formato: HTML (autocontenido, se puede ver en cualquier navegador)",
    },
    "Generate Report": {
        "zh_CN": "生成报告", "zh_TW": "產生報告",
        "ja": "レポートを生成", "ko": "보고서 생성",
        "de": "Bericht erstellen", "fr": "Générer le rapport",
        "es": "Generar informe",
    },
    "FRAME RANGE": {
        "zh_CN": "帧范围", "zh_TW": "影格範圍",
        "ja": "フレーム範囲", "ko": "프레임 범위",
        "de": "BILDBEREICH", "fr": "PLAGE D'IMAGES",
        "es": "RANGO DE FOTOGRAMAS",
    },
    "All frames": {
        "zh_CN": "所有帧", "zh_TW": "所有影格",
        "ja": "すべてのフレーム", "ko": "모든 프레임",
        "de": "Alle Bilder", "fr": "Toutes les images",
        "es": "Todos los fotogramas",
    },
    "From": {
        "zh_CN": "从", "zh_TW": "從",
        "ja": "開始", "ko": "시작",
        "de": "Von", "fr": "De", "es": "Desde",
    },
    "to": {
        "zh_CN": "到", "zh_TW": "到",
        "ja": "まで", "ko": "끝",
        "de": "bis", "fr": "à", "es": "a",
    },
    "Select Output Folder": {
        "zh_CN": "选择输出文件夹", "zh_TW": "選擇輸出資料夾",
        "ja": "出力フォルダーを選択", "ko": "출력 폴더 선택",
        "de": "Ausgabeordner wählen",
        "fr": "Sélectionner le dossier de sortie",
        "es": "Seleccionar carpeta de salida",
    },
    "Exported %1 files → %2": {
        "zh_CN": "已导出 %1 个文件 → %2",
        "zh_TW": "已匯出 %1 個檔案 → %2",
        "ja":    "%1 個のファイルをエクスポートしました → %2",
        "ko":    "%1 개 파일 내보냄 → %2",
        "de":    "%1 Dateien exportiert → %2",
        "fr":    "%1 fichiers exportés → %2",
        "es":    "Exportados %1 archivos → %2",
    },
    "Error: %1": {
        "zh_CN": "错误：%1", "zh_TW": "錯誤：%1",
        "ja": "エラー：%1", "ko": "오류: %1",
        "de": "Fehler: %1", "fr": "Erreur : %1", "es": "Error: %1",
    },
    "Starting…": {
        "zh_CN": "开始中…", "zh_TW": "開始中…",
        "ja": "開始中…", "ko": "시작 중…",
        "de": "Wird gestartet…", "fr": "Démarrage…",
        "es": "Iniciando…",
    },
    "Rendering %1 (%2/%3)": {
        "zh_CN": "正在渲染 %1 (%2/%3)",
        "zh_TW": "正在繪製 %1 (%2/%3)",
        "ja":    "%1 を描画中 (%2/%3)",
        "ko":    "%1 렌더링 중 (%2/%3)",
        "de":    "Rendere %1 (%2/%3)",
        "fr":    "Rendu de %1 (%2/%3)",
        "es":    "Renderizando %1 (%2/%3)",
    },
    "Frame %1/%2": {
        "zh_CN": "帧 %1/%2", "zh_TW": "影格 %1/%2",
        "ja": "フレーム %1/%2", "ko": "프레임 %1/%2",
        "de": "Bild %1/%2", "fr": "Image %1/%2", "es": "Fotograma %1/%2",
    },
    "Exported %1 images → %2": {
        "zh_CN": "已导出 %1 张图像 → %2",
        "zh_TW": "已匯出 %1 張影像 → %2",
        "ja":    "%1 枚の画像をエクスポートしました → %2",
        "ko":    "%1 개 이미지 내보냄 → %2",
        "de":    "%1 Bilder exportiert → %2",
        "fr":    "%1 images exportées → %2",
        "es":    "%1 imágenes exportadas → %2",
    },
    "Report saved → %1": {
        "zh_CN": "报告已保存 → %1",
        "zh_TW": "報告已儲存 → %1",
        "ja":    "レポートを保存しました → %1",
        "ko":    "보고서 저장됨 → %1",
        "de":    "Bericht gespeichert → %1",
        "fr":    "Rapport enregistré → %1",
        "es":    "Informe guardado → %1",
    },

    # ========== FrameNavigator / StrainNavigator ==========
    "FRAME %1/%2": {
        "zh_CN": "帧 %1/%2", "zh_TW": "影格 %1/%2",
        "ja": "フレーム %1/%2", "ko": "프레임 %1/%2",
        "de": "BILD %1/%2", "fr": "IMAGE %1/%2", "es": "FOTOGRAMA %1/%2",
    },

    # ========== ImageList ==========
    "#": {   # frame-index column — usually left as-is
        "zh_CN": "#", "zh_TW": "#", "ja": "#", "ko": "#",
        "de": "#", "fr": "#", "es": "#",
    },
    "Filename": {
        "zh_CN": "文件名", "zh_TW": "檔名",
        "ja": "ファイル名", "ko": "파일 이름",
        "de": "Dateiname", "fr": "Nom de fichier", "es": "Nombre de archivo",
    },
    "Region": {
        "zh_CN": "区域", "zh_TW": "區域",
        "ja": "領域", "ko": "영역",
        "de": "Bereich", "fr": "Région", "es": "Región",
    },
    "Clear Region of Interest": {
        "zh_CN": "清除感兴趣区域",
        "zh_TW": "清除感興趣區域",
        "ja":    "関心領域をクリア",
        "ko":    "관심 영역 지우기",
        "de":    "Region of Interest löschen",
        "fr":    "Effacer la région d'intérêt",
        "es":    "Borrar región de interés",
    },
    "Clear Region of Interest (%1 with region)": {
        "zh_CN": "清除感兴趣区域（%1 帧已有区域）",
        "zh_TW": "清除感興趣區域（%1 影格已有區域）",
        "ja":    "関心領域をクリア（%1 フレームに領域あり）",
        "ko":    "관심 영역 지우기 (%1개 프레임에 영역 있음)",
        "de":    "Region of Interest löschen (%1 mit Region)",
        "fr":    "Effacer la région d'intérêt (%1 avec région)",
        "es":    "Borrar región de interés (%1 con región)",
    },
    "Images": {
        "zh_CN": "图像", "zh_TW": "影像",
        "ja": "画像", "ko": "이미지",
        "de": "Bilder", "fr": "Images", "es": "Imágenes",
    },
    "All Files": {
        "zh_CN": "所有文件", "zh_TW": "所有檔案",
        "ja": "すべてのファイル", "ko": "모든 파일",
        "de": "Alle Dateien", "fr": "Tous les fichiers",
        "es": "Todos los archivos",
    },
    "Selected %1 files for %2 frames — count must match": {
        "zh_CN": "已选择 %1 个文件用于 %2 帧 — 数量必须匹配",
        "zh_TW": "已選擇 %1 個檔案用於 %2 影格 — 數量必須相符",
        "ja":    "%2 フレームに対し %1 個のファイルが選択されました — 数量が一致する必要があります",
        "ko":    "%2 프레임에 대해 %1 개 파일 선택됨 — 개수가 일치해야 합니다",
        "de":    "%1 Dateien für %2 Bilder ausgewählt — Anzahl muss übereinstimmen",
        "fr":    "%1 fichiers sélectionnés pour %2 images — le nombre doit correspondre",
        "es":    "Seleccionados %1 archivos para %2 fotogramas — las cantidades deben coincidir",
    },

    # ========== ParamPanel (refinement levels) ==========
    "Light": {
        "zh_CN": "轻度", "zh_TW": "輕度", "ja": "軽度",
        "ko": "약함", "de": "Leicht", "fr": "Léger", "es": "Ligero",
    },
    "Medium": {
        "zh_CN": "中等", "zh_TW": "中等", "ja": "中程度",
        "ko": "중간", "de": "Mittel", "fr": "Moyen", "es": "Medio",
    },
    "Heavy": {
        "zh_CN": "强", "zh_TW": "強", "ja": "強",
        "ko": "강함", "de": "Stark", "fr": "Fort", "es": "Fuerte",
    },
    "Extra Heavy": {
        "zh_CN": "超强", "zh_TW": "超強", "ja": "最強",
        "ko": "매우 강함", "de": "Sehr stark", "fr": "Très fort",
        "es": "Muy fuerte",
    },
    "Ultra": {
        "zh_CN": "极限", "zh_TW": "極限", "ja": "極限",
        "ko": "극강", "de": "Ultra", "fr": "Ultra", "es": "Ultra",
    },
    "%1 (L%2)": {
        "zh_CN": "%1 (L%2)", "zh_TW": "%1 (L%2)",
        "ja": "%1 (L%2)", "ko": "%1 (L%2)",
        "de": "%1 (L%2)", "fr": "%1 (L%2)", "es": "%1 (L%2)",
    },

    # ========== PhysicalUnitsWidget ==========
    "Pixel size": {
        "zh_CN": "像素尺寸", "zh_TW": "像素尺寸",
        "ja": "ピクセルサイズ", "ko": "픽셀 크기",
        "de": "Pixelgröße", "fr": "Taille de pixel",
        "es": "Tamaño de píxel",
    },
    "Frame rate": {
        "zh_CN": "帧率", "zh_TW": "影格率",
        "ja": "フレームレート", "ko": "프레임 속도",
        "de": "Bildrate", "fr": "Fréquence d'images",
        "es": "Velocidad de fotogramas",
    },
    "Disp: %1  Velocity: %2/s": {
        "zh_CN": "位移：%1  速度：%2/s",
        "zh_TW": "位移：%1  速度：%2/s",
        "ja":    "変位：%1  速度：%2/s",
        "ko":    "변위: %1  속도: %2/s",
        "de":    "Verschiebung: %1  Geschwindigkeit: %2/s",
        "fr":    "Dépl. : %1  Vitesse : %2/s",
        "es":    "Despl.: %1  Velocidad: %2/s",
    },

    # ========== StrainParamPanel ==========
    "VSG (Virtual Strain Gauge) size is the diameter, in pixels, "
    "of the circular region around each mesh node used to fit a "
    "local displacement plane. Strain is then taken as the "
    "plane's slope.\n\n"
    "• Larger VSG → smoother strain, lower spatial resolution.\n"
    "• Smaller VSG → sharper strain, more noise.\n"
    "• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).\n\n"
    "Not used when Method = FEM nodal (there, mesh spacing itself "
    "sets the gauge size).": {
        "zh_CN": "VSG（虚拟应变计，Virtual Strain Gauge）尺寸指围绕每个网格节点、"
                 "用于拟合局部位移平面的圆形区域的直径（像素）。"
                 "应变由该平面的斜率给出。\n\n"
                 "• VSG 越大 → 应变越平滑，空间分辨率越低。\n"
                 "• VSG 越小 → 应变越锐利，但噪声越大。\n"
                 "• 经验法则：VSG ≥ 2 × 子集步长 + 1（默认：41 px）。\n\n"
                 "方法选择 FEM nodal 时不使用此参数（此时由网格间距决定虚拟应变计尺寸）。",
        "zh_TW": "VSG（虛擬應變計，Virtual Strain Gauge）尺寸指圍繞每個網格節點、"
                 "用於擬合局部位移平面的圓形區域的直徑（像素）。"
                 "應變由該平面的斜率給出。\n\n"
                 "• VSG 越大 → 應變越平滑，空間解析度越低。\n"
                 "• VSG 越小 → 應變越銳利，但雜訊越大。\n"
                 "• 經驗法則：VSG ≥ 2 × 子集步長 + 1（預設：41 px）。\n\n"
                 "方法選擇 FEM nodal 時不使用此參數（此時由網格間距決定虛擬應變計尺寸）。",
        "ja":    "VSG（バーチャルひずみゲージ、Virtual Strain Gauge）サイズとは、"
                 "各メッシュノード周辺で局所変位平面をフィットさせるために使う"
                 "円形領域の直径（ピクセル）のことです。ひずみはこの平面の勾配として算出されます。\n\n"
                 "• VSG が大きい → ひずみは平滑になるが、空間解像度は低下。\n"
                 "• VSG が小さい → ひずみは鋭敏になるが、ノイズが増加。\n"
                 "• 目安：VSG ≥ 2 × サブセットステップ + 1（既定：41 px）。\n\n"
                 "方法が FEM nodal の場合は使用されません（そこではメッシュ間隔がゲージサイズを決定します）。",
        "ko":    "VSG(가상 변형률 게이지, Virtual Strain Gauge) 크기는 각 메시 노드 주위에서 "
                 "국소 변위 평면을 피팅하는 데 사용되는 원형 영역의 지름(픽셀)입니다. "
                 "변형률은 이 평면의 기울기로 얻어집니다.\n\n"
                 "• VSG가 클수록 → 변형률이 매끄럽고 공간 해상도가 낮음.\n"
                 "• VSG가 작을수록 → 변형률이 날카롭지만 노이즈 증가.\n"
                 "• 경험 법칙: VSG ≥ 2 × 서브셋 스텝 + 1 (기본값: 41 px).\n\n"
                 "Method = FEM nodal일 때는 사용되지 않습니다(그 경우 메시 간격 자체가 게이지 크기를 결정).",
        "de":    "VSG (Virtual Strain Gauge) ist der Durchmesser in Pixel des kreisförmigen "
                 "Bereichs um jeden Netzknoten, der zum Anpassen einer lokalen Verschiebungs-"
                 "ebene verwendet wird. Die Dehnung ergibt sich aus der Steigung dieser Ebene.\n\n"
                 "• Größeres VSG → glattere Dehnung, geringere räumliche Auflösung.\n"
                 "• Kleineres VSG → schärfere Dehnung, mehr Rauschen.\n"
                 "• Faustregel: VSG ≥ 2 × Subset-Schritt + 1 (Standard: 41 px).\n\n"
                 "Nicht verwendet bei Methode = FEM nodal (dort bestimmt der Netzabstand die Größe).",
        "fr":    "La taille VSG (Virtual Strain Gauge, jauge de déformation virtuelle) est le "
                 "diamètre, en pixels, de la région circulaire autour de chaque nœud du maillage, "
                 "utilisée pour ajuster un plan de déplacement local. La déformation est ensuite "
                 "prise comme la pente de ce plan.\n\n"
                 "• VSG plus grande → déformation plus lisse, résolution spatiale plus faible.\n"
                 "• VSG plus petite → déformation plus fine, mais plus de bruit.\n"
                 "• Règle empirique : VSG ≥ 2 × pas de subset + 1 (par défaut : 41 px).\n\n"
                 "Non utilisée quand Méthode = FEM nodal (l'espacement du maillage fixe alors la taille).",
        "es":    "El tamaño VSG (Virtual Strain Gauge, galga de deformación virtual) es el diámetro, "
                 "en píxeles, de la región circular alrededor de cada nodo de malla utilizada para "
                 "ajustar un plano de desplazamiento local. La deformación se toma como la pendiente "
                 "de dicho plano.\n\n"
                 "• VSG más grande → deformación más suave, menor resolución espacial.\n"
                 "• VSG más pequeño → deformación más nítida, pero con más ruido.\n"
                 "• Regla práctica: VSG ≥ 2 × paso del subset + 1 (predeterminado: 41 px).\n\n"
                 "No se usa con Method = FEM nodal (allí el espaciado de la malla establece el tamaño).",
    },

    "\u26a0 VSG radius (%1 px) < DIC node spacing (%2 px); "
    "plane fit will fail. Use VSG \u2265 %3 px or switch "
    "Method to FEM nodal.": {
        "zh_CN": "⚠ VSG 半径（%1 px）< DIC 节点间距（%2 px）；"
                 "平面拟合将失败。请将 VSG ≥ %3 px 或将方法切换为 FEM nodal。",
        "zh_TW": "⚠ VSG 半徑（%1 px）< DIC 節點間距（%2 px）；"
                 "平面擬合將失敗。請將 VSG ≥ %3 px 或將方法切換為 FEM nodal。",
        "ja":    "⚠ VSG 半径（%1 px）< DIC ノード間隔（%2 px）；"
                 "平面フィットは失敗します。VSG ≥ %3 px にするか、方法を FEM nodal に切り替えてください。",
        "ko":    "⚠ VSG 반경(%1 px) < DIC 노드 간격(%2 px); "
                 "평면 피팅이 실패합니다. VSG ≥ %3 px로 설정하거나 Method를 FEM nodal로 전환하세요.",
        "de":    "⚠ VSG-Radius (%1 px) < DIC-Knotenabstand (%2 px); "
                 "Ebenenanpassung wird fehlschlagen. VSG ≥ %3 px verwenden oder Methode auf FEM nodal wechseln.",
        "fr":    "⚠ Rayon VSG (%1 px) < espacement des nœuds DIC (%2 px) ; "
                 "l'ajustement de plan échouera. Utilisez VSG ≥ %3 px ou passez la Méthode en FEM nodal.",
        "es":    "⚠ Radio VSG (%1 px) < espaciado de nodos DIC (%2 px); "
                 "el ajuste de plano fallará. Use VSG ≥ %3 px o cambie Método a FEM nodal.",
    },

    # ========== StrainVizPanel ==========
    "Deformed": {
        "zh_CN": "变形后", "zh_TW": "變形後",
        "ja": "変形後", "ko": "변형 후",
        "de": "Verformt", "fr": "Déformé", "es": "Deformado",
    },
    "Range": {
        "zh_CN": "范围", "zh_TW": "範圍",
        "ja": "範囲", "ko": "범위",
        "de": "Bereich", "fr": "Plage", "es": "Rango",
    },

    # ========== StrainWindow ==========
    "Strain compute failed: %1: %2": {
        "zh_CN": "应变计算失败：%1：%2",
        "zh_TW": "應變計算失敗：%1：%2",
        "ja":    "ひずみ計算に失敗しました：%1：%2",
        "ko":    "변형률 계산 실패: %1: %2",
        "de":    "Dehnungsberechnung fehlgeschlagen: %1: %2",
        "fr":    "Échec du calcul de déformation : %1 : %2",
        "es":    "Fallo en el cálculo de deformación: %1: %2",
    },
    "Strain compute failed: %1": {
        "zh_CN": "应变计算失败：%1",
        "zh_TW": "應變計算失敗：%1",
        "ja":    "ひずみ計算に失敗しました：%1",
        "ko":    "변형률 계산 실패: %1",
        "de":    "Dehnungsberechnung fehlgeschlagen: %1",
        "fr":    "Échec du calcul de déformation : %1",
        "es":    "Fallo en el cálculo de deformación: %1",
    },
    "Strain Computation Failed": {
        "zh_CN": "应变计算失败",
        "zh_TW": "應變計算失敗",
        "ja":    "ひずみ計算に失敗しました",
        "ko":    "변형률 계산 실패",
        "de":    "Dehnungsberechnung fehlgeschlagen",
        "fr":    "Échec du calcul de déformation",
        "es":    "Fallo en el cálculo de deformación",
    },
    "Strain computation complete.": {
        "zh_CN": "应变计算完成。",
        "zh_TW": "應變計算完成。",
        "ja":    "ひずみ計算が完了しました。",
        "ko":    "변형률 계산 완료.",
        "de":    "Dehnungsberechnung abgeschlossen.",
        "fr":    "Calcul de déformation terminé.",
        "es":    "Cálculo de deformación completado.",
    },
    "Strain window: no displacement results to post-process.": {
        "zh_CN": "应变窗口：没有可后处理的位移结果。",
        "zh_TW": "應變視窗：沒有可後處理的位移結果。",
        "ja":    "ひずみウィンドウ：後処理する変位結果がありません。",
        "ko":    "변형률 창: 후처리할 변위 결과가 없습니다.",
        "de":    "Dehnungsfenster: Keine Verschiebungs-Ergebnisse zur Nachbearbeitung.",
        "fr":    "Fenêtre de déformation : aucun résultat de déplacement à post-traiter.",
        "es":    "Ventana de deformación: no hay resultados de desplazamiento para posprocesar.",
    },

    # ========== App (main window ROI import / strain window gate) ==========
    "Run DIC first -- no displacement results to post-process.": {
        "zh_CN": "请先运行 DIC —— 当前没有可后处理的位移结果。",
        "zh_TW": "請先執行 DIC —— 目前沒有可後處理的位移結果。",
        "ja":    "先に DIC を実行してください —— 後処理する変位結果がありません。",
        "ko":    "DIC를 먼저 실행하세요 —— 후처리할 변위 결과가 없습니다.",
        "de":    "DIC zuerst ausführen — keine Verschiebungs-Ergebnisse zur Nachbearbeitung.",
        "fr":    "Exécutez d'abord le DIC — aucun résultat de déplacement à post-traiter.",
        "es":    "Ejecute primero el DIC — no hay resultados de desplazamiento para posprocesar.",
    },

    # ========== PipelineController (start() + _on_finished()) ==========
    "  Loaded %1 images, shape=%2": {
        "zh_CN": "  已加载 %1 张图像，尺寸=%2",
        "zh_TW": "  已載入 %1 張影像，尺寸=%2",
        "ja":    "  %1 枚の画像を読み込みました、shape=%2",
        "ko":    "  %1 개 이미지 로드됨, shape=%2",
        "de":    "  %1 Bilder geladen, Form=%2",
        "fr":    "  %1 images chargées, forme=%2",
        "es":    "  %1 imágenes cargadas, forma=%2",
    },
    "  ROI mask: %1, %2 pixels (%3%)": {
        "zh_CN": "  感兴趣区域蒙版：%1，%2 像素（%3%）",
        "zh_TW": "  感興趣區域遮罩：%1，%2 像素（%3%）",
        "ja":    "  ROI マスク：%1、%2 ピクセル（%3%）",
        "ko":    "  ROI 마스크: %1, %2 픽셀 (%3%)",
        "de":    "  ROI-Maske: %1, %2 Pixel (%3%)",
        "fr":    "  Masque ROI : %1, %2 pixels (%3%)",
        "es":    "  Máscara ROI: %1, %2 píxeles (%3%)",
    },
    "Run cancelled: define per-frame Regions of Interest "
    "for the missing reference frames or accept the "
    "inherited frame-1 mask in the next run.": {
        "zh_CN": "已取消运行：请为缺失的参考帧定义逐帧感兴趣区域，"
                 "或在下次运行时接受继承自第 1 帧的蒙版。",
        "zh_TW": "已取消執行：請為缺失的參考影格定義逐影格感興趣區域，"
                 "或在下次執行時接受繼承自第 1 影格的遮罩。",
        "ja":    "実行をキャンセルしました：欠けている参照フレームに対して"
                 "フレーム別の関心領域を定義するか、次回実行時に"
                 "第 1 フレームのマスクを継承してください。",
        "ko":    "실행 취소됨: 누락된 참조 프레임에 대해 프레임별 관심 영역을 "
                 "정의하거나, 다음 실행 시 프레임 1의 마스크를 그대로 사용하도록 허용하세요.",
        "de":    "Lauf abgebrochen: Definieren Sie pro Bild Regions of Interest "
                 "für die fehlenden Referenzbilder, oder akzeptieren Sie beim "
                 "nächsten Lauf die vom 1. Bild geerbte Maske.",
        "fr":    "Exécution annulée : définissez les régions d'intérêt par image "
                 "pour les images de référence manquantes, ou acceptez le "
                 "masque hérité de l'image 1 au prochain lancement.",
        "es":    "Ejecución cancelada: defina regiones de interés por fotograma "
                 "para los fotogramas de referencia que faltan, o acepte la "
                 "máscara heredada del fotograma 1 en la próxima ejecución.",
    },

    # ========== PipelineWorker.run() ==========
    "Starting DIC analysis...": {
        "zh_CN": "开始 DIC 分析…",
        "zh_TW": "開始 DIC 分析…",
        "ja":    "DIC 解析を開始します…",
        "ko":    "DIC 분석 시작 중…",
        "de":    "DIC-Analyse wird gestartet…",
        "fr":    "Démarrage de l'analyse DIC…",
        "es":    "Iniciando análisis DIC…",
    },
    "Analysis complete in %1s": {
        "zh_CN": "分析完成，用时 %1 秒",
        "zh_TW": "分析完成，耗時 %1 秒",
        "ja":    "解析が完了しました（%1 秒）",
        "ko":    "분석 완료 (%1초)",
        "de":    "Analyse in %1 s abgeschlossen",
        "fr":    "Analyse terminée en %1 s",
        "es":    "Análisis completado en %1 s",
    },
    "Analysis stopped by user.": {
        "zh_CN": "用户已停止分析。",
        "zh_TW": "使用者已停止分析。",
        "ja":    "ユーザーにより解析が停止されました。",
        "ko":    "사용자가 분석을 중지했습니다.",
        "de":    "Analyse wurde vom Benutzer gestoppt.",
        "fr":    "Analyse arrêtée par l'utilisateur.",
        "es":    "Análisis detenido por el usuario.",
    },

    # ========== StrainParamPanel smoothing presets ==========
    "Off": {
        "zh_CN": "关闭", "zh_TW": "關閉",
        "ja": "オフ", "ko": "끔",
        "de": "Aus", "fr": "Désactivé", "es": "Desactivado",
    },
    "Light (σ = 0.5 × step)": {
        "zh_CN": "轻度（σ = 0.5 × step）",
        "zh_TW": "輕度（σ = 0.5 × step）",
        "ja":    "軽度（σ = 0.5 × step）",
        "ko":    "약함 (σ = 0.5 × step)",
        "de":    "Leicht (σ = 0,5 × step)",
        "fr":    "Léger (σ = 0,5 × step)",
        "es":    "Ligero (σ = 0,5 × step)",
    },
    "Medium (σ = 1 × step)": {
        "zh_CN": "中等（σ = 1 × step）",
        "zh_TW": "中等（σ = 1 × step）",
        "ja":    "中程度（σ = 1 × step）",
        "ko":    "중간 (σ = 1 × step)",
        "de":    "Mittel (σ = 1 × step)",
        "fr":    "Moyen (σ = 1 × step)",
        "es":    "Medio (σ = 1 × step)",
    },
    "Strong (σ = 2 × step) ⚠": {
        "zh_CN": "强（σ = 2 × step）⚠",
        "zh_TW": "強（σ = 2 × step）⚠",
        "ja":    "強（σ = 2 × step）⚠",
        "ko":    "강함 (σ = 2 × step) ⚠",
        "de":    "Stark (σ = 2 × step) ⚠",
        "fr":    "Fort (σ = 2 × step) ⚠",
        "es":    "Fuerte (σ = 2 × step) ⚠",
    },

    # ========== BatchImportDialog — size pre-scan & 1:N assignment ==========

    # ========== AutoFixedSelector — explicit Auto/Fixed color-range mode ====
    "Fixed": {
        "zh_CN": "固定", "zh_TW": "固定",
        "ja": "固定", "ko": "고정",
        "de": "Fest", "fr": "Fixe", "es": "Fijo",
    },
    "Rescale the color range to each frame's data range": {
        "zh_CN": "根据每帧的数据范围自动缩放颜色范围",
        "zh_TW": "根據每幀的資料範圍自動縮放顏色範圍",
        "ja":    "各フレームのデータ範囲に合わせてカラーレンジを再スケールします",
        "ko":    "각 프레임의 데이터 범위에 맞춰 색상 범위를 다시 조정합니다",
        "de":    "Farbbereich an den Datenbereich jedes Frames anpassen",
        "fr":    "Ajuster la plage de couleurs à la plage de données de chaque image",
        "es":    "Ajustar el rango de colores al rango de datos de cada fotograma",
    },
    "Keep the manual Min/Max bounds for every frame": {
        "zh_CN": "所有帧都使用手动设置的最小/最大值",
        "zh_TW": "所有幀都使用手動設定的最小/最大值",
        "ja":    "すべてのフレームで手動の最小/最大値を使用します",
        "ko":    "모든 프레임에서 수동 최소/최대 값을 유지합니다",
        "de":    "Manuelle Min/Max-Grenzen für alle Frames beibehalten",
        "fr":    "Conserver les bornes Min/Max manuelles pour toutes les images",
        "es":    "Mantener los límites Mín/Máx manuales en todos los fotogramas",
    },
    # "Range" label reused by the ExportDialog preview tab
    # ===== Display mode: geometry and background chosen separately =========
    "Deformed frame": {
        "zh_CN": "变形帧",
        "zh_TW": "變形幀",
        "ja":    "変形フレーム",
        "ko":    "변형된 프레임",
        "de":    "Verformtes Bild",
        "fr":    "Image déformée",
        "es":    "Fotograma deformado",
    },
    "Reference frame": {
        "zh_CN": "参考帧",
        "zh_TW": "參考幀",
        "ja":    "参照フレーム",
        "ko":    "참조 프레임",
        "de":    "Referenzbild",
        "fr":    "Image de référence",
        "es":    "Fotograma de referencia",
    },
    "Show on": {
        "zh_CN": "显示于",
        "zh_TW": "顯示於",
        "ja":    "表示先",
        "ko":    "표시 기준",
        "de":    "Anzeigen auf",
        "fr":    "Afficher sur",
        "es":    "Mostrar en",
    },
    "Show background image": {
        "zh_CN": "显示背景图像",
        "zh_TW": "顯示背景影像",
        "ja":    "背景画像を表示",
        "ko":    "배경 이미지 표시",
        "de":    "Hintergrundbild anzeigen",
        "fr":    "Afficher l'image de fond",
        "es":    "Mostrar imagen de fondo",
    },
    "Transparent": {
        "zh_CN": "透明",
        "zh_TW": "透明",
        "ja":    "透明",
        "ko":    "투명",
        "de":    "Transparent",
        "fr":    "Transparent",
        "es":    "Transparente",
    },
    "Hidden background": {
        "zh_CN": "隐藏背景填充",
        "zh_TW": "隱藏背景填充",
        "ja":    "非表示時の背景",
        "ko":    "숨김 시 배경",
        "de":    "Ausgeblendeter Hintergrund",
        "fr":    "Arrière-plan masqué",
        "es":    "Fondo oculto",
    },
    "Plot the field at the deformed node positions, or at their positions in the reference frame.": {
        "zh_CN": "将字段绘制在变形后的节点位置，或绘制在其在参考帧中的位置。",
        "zh_TW": "將欄位繪製在變形後的節點位置，或繪製在其於參考幀中的位置。",
        "ja":    "フィールドを変形後のノード位置に描画するか、参照フレームでの位置に描画します。",
        "ko":    "필드를 변형된 노드 위치에 그리거나 참조 프레임에서의 위치에 그립니다.",
        "de":    "Das Feld an den verformten Knotenpositionen zeichnen oder an ihren Positionen im Referenzbild.",
        "fr":    "Tracer le champ aux positions déformées des nœuds, ou à leurs positions dans l'image de référence.",
        "es":    "Dibujar el campo en las posiciones deformadas de los nodos, o en sus posiciones en el fotograma de referencia.",
    },
    "Uncheck to show the field on its own, with no speckle image behind it.": {
        "zh_CN": "取消勾选可仅显示字段，其后不显示散斑图像。",
        "zh_TW": "取消勾選可僅顯示欄位，其後不顯示散斑影像。",
        "ja":    "チェックを外すと、背後にスペックル画像を表示せずフィールドのみを表示します。",
        "ko":    "선택을 해제하면 뒤에 스페클 이미지 없이 필드만 표시합니다.",
        "de":    "Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter anzuzeigen.",
        "fr":    "Décochez pour n'afficher que le champ, sans image de mouchetis derrière.",
        "es":    "Desmarque para mostrar solo el campo, sin imagen de moteado detrás.",
    },
    "Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview & Colorbar tab.": {
        "zh_CN": "取消勾选可仅导出字段，其后不含散斑图像。填充色在 Preview & Colorbar 页选择。",
        "zh_TW": "取消勾選可僅匯出欄位，其後不含散斑影像。填充色於 Preview & Colorbar 頁選擇。",
        "ja":    "チェックを外すと、背後にスペックル画像を含めずフィールドのみを書き出します。塗りつぶしは Preview & Colorbar タブで選択します。",
        "ko":    "선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. 채우기는 Preview & Colorbar 탭에서 선택합니다.",
        "de":    "Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter zu exportieren. Die Füllung wird im Reiter \u201ePreview & Colorbar\u201c gewählt.",
        "fr":    "Décochez pour n'exporter que le champ, sans image de mouchetis derrière. Le remplissage se choisit dans l'onglet Preview & Colorbar.",
        "es":    "Desmarque para exportar solo el campo, sin imagen de moteado detrás. El relleno se elige en la pestaña Preview & Colorbar.",
    },
    "Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.": {
        "zh_CN": "取消勾选可仅导出字段，其后不含散斑图像。GIF 和 MP4 无法保存透明度，透明填充将写为白色。",
        "zh_TW": "取消勾選可僅匯出欄位，其後不含散斑影像。GIF 與 MP4 無法儲存透明度，透明填充將寫為白色。",
        "ja":    "チェックを外すと、背後にスペックル画像を含めずフィールドのみを書き出します。GIF と MP4 は透明度を保存できないため、透明の塗りつぶしは白として書き出されます。",
        "ko":    "선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. GIF와 MP4는 투명도를 저장할 수 없으므로 투명 채우기는 흰색으로 기록됩니다.",
        "de":    "Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter zu exportieren. GIF und MP4 können keine Transparenz speichern; eine transparente Füllung wird als Weiß geschrieben.",
        "fr":    "Décochez pour n'exporter que le champ, sans image de mouchetis derrière. GIF et MP4 ne peuvent pas stocker la transparence : un remplissage transparent est écrit en blanc.",
        "es":    "Desmarque para exportar solo el campo, sin imagen de moteado detrás. GIF y MP4 no pueden almacenar transparencia, por lo que un relleno transparente se escribe como blanco.",
    },
    "Deformed: the field is drawn at the displaced node positions (reference + displacement), over each frame's own photo.\nReference: drawn at the original node positions, over the first frame.": {
        "zh_CN": "变形帧：字段绘制在位移后的节点位置（参考位置 + 位移），叠加在每一帧自己的照片上。\n参考帧：绘制在原始节点位置，叠加在第一帧上。",
        "zh_TW": "變形幀：欄位繪製在位移後的節點位置（參考位置 + 位移），疊加在每一幀自己的照片上。\n參考幀：繪製在原始節點位置，疊加在第一幀上。",
        "ja":    "変形フレーム：フィールドを変位後のノード位置（参照位置 + 変位）に描画し、各フレーム自身の写真に重ねます。\n参照フレーム：元のノード位置に描画し、最初のフレームに重ねます。",
        "ko":    "변형된 프레임: 필드를 변위된 노드 위치(참조 위치 + 변위)에 그려 각 프레임 자체의 사진 위에 겹칩니다.\n참조 프레임: 원래 노드 위치에 그려 첫 번째 프레임 위에 겹칩니다.",
        "de":    "Verformtes Bild: Das Feld wird an den verschobenen Knotenpositionen (Referenz + Verschiebung) über dem jeweils eigenen Foto jedes Bildes gezeichnet.\nReferenzbild: an den ursprünglichen Knotenpositionen über dem ersten Bild gezeichnet.",
        "fr":    "Image déformée : le champ est tracé aux positions déplacées des nœuds (référence + déplacement), par-dessus la photo propre à chaque image.\nImage de référence : tracé aux positions d'origine des nœuds, par-dessus la première image.",
        "es":    "Fotograma deformado: el campo se dibuja en las posiciones desplazadas de los nodos (referencia + desplazamiento), sobre la foto propia de cada fotograma.\nFotograma de referencia: se dibuja en las posiciones originales de los nodos, sobre el primer fotograma.",
    },
    "Fill used where the background image would have been, when \'Show background image' is off.\nTransparency is kept for PNG and TIFF; JPEG, GIF and MP4 have no alpha channel and get white instead.": {
        "zh_CN": "关闭“显示背景图像”时，用于填充原本显示背景图像的区域。\nPNG 和 TIFF 会保留透明度；JPEG、GIF 和 MP4 没有 alpha 通道，将改用白色。",
        "zh_TW": "關閉「顯示背景影像」時，用於填充原本顯示背景影像的區域。\nPNG 與 TIFF 會保留透明度；JPEG、GIF 和 MP4 沒有 alpha 通道，將改用白色。",
        "ja":    "「背景画像を表示」をオフにしたとき、背景画像があった領域を塗りつぶす色です。\nPNG と TIFF では透明度が保持されます。JPEG、GIF、MP4 にはアルファチャンネルがないため、白になります。",
        "ko":    "‘배경 이미지 표시’를 끄면 배경 이미지가 있던 자리를 채우는 색입니다.\nPNG와 TIFF는 투명도를 유지합니다. JPEG, GIF, MP4는 알파 채널이 없어 흰색으로 대체됩니다.",
        "de":    "Füllung für den Bereich, in dem sonst das Hintergrundbild läge, wenn \u201eHintergrundbild anzeigen\u201c aus ist.\nBei PNG und TIFF bleibt die Transparenz erhalten; JPEG, GIF und MP4 haben keinen Alphakanal und erhalten stattdessen Weiß.",
        "fr":    "Remplissage utilisé là où se serait trouvée l'image de fond, quand « Afficher l'image de fond » est décoché.\nLa transparence est conservée pour PNG et TIFF ; JPEG, GIF et MP4 n'ont pas de canal alpha et reçoivent du blanc.",
        "es":    "Relleno utilizado donde habría estado la imagen de fondo, cuando «Mostrar imagen de fondo» está desactivado.\nLa transparencia se conserva para PNG y TIFF; JPEG, GIF y MP4 no tienen canal alfa y reciben blanco.",
    },
    "What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.": {
        "zh_CN": "背景隐藏时用什么替代图像。导出时 PNG 和 TIFF 会保留透明度，其他格式改用白色。",
        "zh_TW": "背景隱藏時用什麼替代影像。匯出時 PNG 與 TIFF 會保留透明度，其他格式改用白色。",
        "ja":    "背景を非表示にしたとき、画像の代わりに何を表示するかです。書き出しでは PNG と TIFF が透明度を保持し、他の形式は白になります。",
        "ko":    "배경을 숨겼을 때 이미지를 대신할 채우기입니다. 내보낼 때 PNG와 TIFF는 투명도를 유지하며, 다른 형식은 흰색이 됩니다.",
        "de":    "Was das Bild ersetzt, wenn es ausgeblendet ist. Beim Export bleibt die Transparenz bei PNG und TIFF erhalten; andere Formate erhalten Weiß.",
        "fr":    "Ce qui remplace l'image lorsqu'elle est masquée. À l'export, la transparence est conservée pour PNG et TIFF ; les autres formats reçoivent du blanc.",
        "es":    "Lo que reemplaza a la imagen cuando está oculta. Al exportar, la transparencia se conserva para PNG y TIFF; los demás formatos reciben blanco.",
    },
    # ===== Moved from NUMERUS_TRANSLATIONS (misfiled as plurals) =====
    "Analysis": {
        "zh_CN": "分析", "zh_TW": "分析", "ja": "解析", "ko": "분석",
        "de": "Analyse", "fr": "Analyse", "es": "Análisis",
    },
    "Strain Field": {
        "zh_CN": "应变场", "zh_TW": "應變場", "ja": "ひずみ場", "ko": "변형률 장",
        "de": "Dehnungsfeld", "fr": "Champ de déformation",
        "es": "Campo de deformación",
    },
    "Point": {
        "zh_CN": "点", "zh_TW": "點", "ja": "点", "ko": "점",
        "de": "Punkt", "fr": "Point", "es": "Punto",
    },
    "Line": {
        "zh_CN": "线段", "zh_TW": "線段", "ja": "線分", "ko": "선분",
        "de": "Linie", "fr": "Ligne", "es": "Línea",
    },
    "Rectangle": {
        "zh_CN": "矩形", "zh_TW": "矩形", "ja": "矩形", "ko": "사각형",
        "de": "Rechteck", "fr": "Rectangle", "es": "Rectángulo",
    },
    "Circle": {
        "zh_CN": "圆形", "zh_TW": "圓形", "ja": "円", "ko": "원",
        "de": "Kreis", "fr": "Cercle", "es": "Círculo",
    },
    "Polygon": {
        "zh_CN": "多边形", "zh_TW": "多邊形", "ja": "多角形", "ko": "다각형",
        "de": "Polygon", "fr": "Polygone", "es": "Polígono",
    },
    "Click once to place a point probe.": {
        "zh_CN": "单击一次放置点探针。",
        "zh_TW": "按一下放置點探針。",
        "ja": "1 回クリックして点プローブを配置します。",
        "ko": "한 번 클릭하여 점 프로브를 배치합니다.",
        "de": "Einmal klicken, um eine Punktsonde zu setzen.",
        "fr": "Cliquez une fois pour placer une sonde ponctuelle.",
        "es": "Haga clic una vez para colocar una sonda puntual.",
    },
    "Click twice: start and end of the gauge.": {
        "zh_CN": "单击两次：标距的起点和终点。",
        "zh_TW": "按兩下：標距的起點與終點。",
        "ja": "2 回クリック：標点間の始点と終点。",
        "ko": "두 번 클릭: 게이지의 시작점과 끝점.",
        "de": "Zweimal klicken: Anfang und Ende der Messlänge.",
        "fr": "Cliquez deux fois : début et fin de la base de mesure.",
        "es": "Haga clic dos veces: inicio y fin de la base de medida.",
    },
    "Click twice: opposite corners.": {
        "zh_CN": "单击两次：对角两点。",
        "zh_TW": "按兩下：對角兩點。",
        "ja": "2 回クリック：対角の 2 点。",
        "ko": "두 번 클릭: 마주 보는 두 모서리.",
        "de": "Zweimal klicken: gegenüberliegende Ecken.",
        "fr": "Cliquez deux fois : coins opposés.",
        "es": "Haga clic dos veces: esquinas opuestas.",
    },
    "Click twice: centre, then the edge.": {
        "zh_CN": "单击两次：先圆心，后边缘。",
        "zh_TW": "按兩下：先圓心，後邊緣。",
        "ja": "2 回クリック：中心、次に円周。",
        "ko": "두 번 클릭: 중심, 그다음 가장자리.",
        "de": "Zweimal klicken: Mittelpunkt, dann Rand.",
        "fr": "Cliquez deux fois : centre, puis bord.",
        "es": "Haga clic dos veces: centro y luego borde.",
    },
    "Click each vertex, then double-click to close.": {
        "zh_CN": "逐个单击顶点，双击闭合。",
        "zh_TW": "逐一按下頂點，按兩下閉合。",
        "ja": "各頂点をクリックし、ダブルクリックで閉じます。",
        "ko": "각 꼭짓점을 클릭한 뒤 두 번 클릭하여 닫습니다.",
        "de": "Jeden Eckpunkt anklicken, dann per Doppelklick schließen.",
        "fr": "Cliquez sur chaque sommet, puis double-cliquez pour fermer.",
        "es": "Haga clic en cada vértice y doble clic para cerrar.",
    },
    "Esc cancels placement": {
        "zh_CN": "Esc 取消放置",
        "zh_TW": "Esc 取消放置",
        "ja": "Esc で配置をキャンセル",
        "ko": "Esc 키로 배치 취소",
        "de": "Esc bricht das Setzen ab",
        "fr": "Esc annule le placement",
        "es": "Esc cancela la colocación",
    },
    "Show": {
        "zh_CN": "显示", "zh_TW": "顯示", "ja": "表示", "ko": "표시",
        "de": "Anzeigen", "fr": "Afficher", "es": "Mostrar",
    },
    "Name": {
        "zh_CN": "名称", "zh_TW": "名稱", "ja": "名前", "ko": "이름",
        "de": "Name", "fr": "Nom", "es": "Nombre",
    },
    "Type": {
        "zh_CN": "类型", "zh_TW": "類型", "ja": "種類", "ko": "종류",
        "de": "Typ", "fr": "Type", "es": "Tipo",
    },
    "Colour": {
        "zh_CN": "颜色", "zh_TW": "顏色", "ja": "色", "ko": "색상",
        "de": "Farbe", "fr": "Couleur", "es": "Color",
    },
    "Colour…": {
        "zh_CN": "颜色…", "zh_TW": "顏色…", "ja": "色…", "ko": "색상…",
        "de": "Farbe…", "fr": "Couleur…", "es": "Color…",
    },
    "Delete": {
        "zh_CN": "删除", "zh_TW": "刪除", "ja": "削除", "ko": "삭제",
        "de": "Löschen", "fr": "Supprimer", "es": "Eliminar",
    },
    "Clear All": {
        "zh_CN": "全部清除", "zh_TW": "全部清除", "ja": "すべて消去",
        "ko": "모두 지우기", "de": "Alle löschen", "fr": "Tout effacer",
        "es": "Borrar todo",
    },
    "Clear All Probes": {
        "zh_CN": "清除所有探针", "zh_TW": "清除所有探針",
        "ja": "すべてのプローブを消去", "ko": "모든 프로브 지우기",
        "de": "Alle Sonden löschen", "fr": "Effacer toutes les sondes",
        "es": "Borrar todas las sondas",
    },
    "Delete every probe? This cannot be undone.": {
        "zh_CN": "删除所有探针？此操作无法撤销。",
        "zh_TW": "刪除所有探針？此操作無法復原。",
        "ja": "すべてのプローブを削除しますか？この操作は取り消せません。",
        "ko": "모든 프로브를 삭제하시겠습니까? 되돌릴 수 없습니다.",
        "de": "Alle Sonden löschen? Das lässt sich nicht rückgängig machen.",
        "fr": "Supprimer toutes les sondes ? Cette action est irréversible.",
        "es": "¿Eliminar todas las sondas? Esta acción no se puede deshacer.",
    },
    "Added probe '%1'.": {
        "zh_CN": "已添加探针「%1」。",
        "zh_TW": "已新增探針「%1」。",
        "ja": "プローブ「%1」を追加しました。",
        "ko": "프로브 '%1'을(를) 추가했습니다.",
        "de": "Sonde '%1' hinzugefügt.",
        "fr": "Sonde « %1 » ajoutée.",
        "es": "Sonda «%1» añadida.",
    },
    "Compare:": {
        "zh_CN": "比较：", "zh_TW": "比較：", "ja": "比較：", "ko": "비교:",
        "de": "Vergleichen:", "fr": "Comparer :", "es": "Comparar:",
    },
    "Field:": {
        "zh_CN": "字段：", "zh_TW": "欄位：", "ja": "フィールド：", "ko": "필드:",
        "de": "Feld:", "fr": "Champ :", "es": "Campo:",
    },
    "Statistic:": {
        "zh_CN": "统计量：", "zh_TW": "統計量：", "ja": "統計量：", "ko": "통계량:",
        "de": "Statistik:", "fr": "Statistique :", "es": "Estadístico:",
    },
    "Point probes": {
        "zh_CN": "点探针", "zh_TW": "點探針", "ja": "点プローブ", "ko": "점 프로브",
        "de": "Punktsonden", "fr": "Sondes ponctuelles", "es": "Sondas puntuales",
    },
    "Line probes": {
        "zh_CN": "线探针", "zh_TW": "線探針", "ja": "線プローブ", "ko": "선 프로브",
        "de": "Liniensonden", "fr": "Sondes linéaires", "es": "Sondas lineales",
    },
    "Region probes": {
        "zh_CN": "区域探针", "zh_TW": "區域探針", "ja": "領域プローブ",
        "ko": "영역 프로브", "de": "Bereichssonden",
        "fr": "Sondes de région", "es": "Sondas de región",
    },
    "Value": {
        "zh_CN": "数值", "zh_TW": "數值", "ja": "値", "ko": "값",
        "de": "Wert", "fr": "Valeur", "es": "Valor",
    },
    "Mean": {
        "zh_CN": "平均值", "zh_TW": "平均值", "ja": "平均", "ko": "평균",
        "de": "Mittelwert", "fr": "Moyenne", "es": "Media",
    },
    "Median": {
        "zh_CN": "中位数", "zh_TW": "中位數", "ja": "中央値", "ko": "중앙값",
        "de": "Median", "fr": "Médiane", "es": "Mediana",
    },
    "Maximum": {
        "zh_CN": "最大值", "zh_TW": "最大值", "ja": "最大値", "ko": "최댓값",
        "de": "Maximum", "fr": "Maximum", "es": "Máximo",
    },
    "Minimum": {
        "zh_CN": "最小值", "zh_TW": "最小值", "ja": "最小値", "ko": "최솟값",
        "de": "Minimum", "fr": "Minimum", "es": "Mínimo",
    },
    "Standard deviation": {
        "zh_CN": "标准差", "zh_TW": "標準差", "ja": "標準偏差", "ko": "표준편차",
        "de": "Standardabweichung", "fr": "Écart-type", "es": "Desviación típica",
    },
    "Valid fraction": {
        "zh_CN": "有效比例", "zh_TW": "有效比例", "ja": "有効割合",
        "ko": "유효 비율", "de": "Gültiger Anteil", "fr": "Fraction valide",
        "es": "Fracción válida",
    },
    "Engineering strain": {
        "zh_CN": "工程应变", "zh_TW": "工程應變", "ja": "公称ひずみ",
        "ko": "공칭 변형률", "de": "Technische Dehnung",
        "fr": "Déformation nominale", "es": "Deformación ingenieril",
    },
    "Crack opening": {
        "zh_CN": "裂纹张开位移", "zh_TW": "裂紋張開位移", "ja": "き裂開口変位",
        "ko": "균열 개구 변위", "de": "Rissöffnung",
        "fr": "Ouverture de fissure", "es": "Apertura de grieta",
    },
    "Minimum valid fraction:": {
        "zh_CN": "最低有效比例：", "zh_TW": "最低有效比例：",
        "ja": "最小有効割合：", "ko": "최소 유효 비율:",
        "de": "Mindestanteil gültiger Punkte:",
        "fr": "Fraction valide minimale :",
        "es": "Fracción válida mínima:",
    },
    "A frame is left blank when fewer than this fraction of the probe's points are reliable. Guards against a curve that stays smooth while its sample shrinks away.": {
        "zh_CN": "当探针中可靠点的比例低于此值时，该帧留空。用于防止曲线在样本量悄悄塌陷时依然平滑。",
        "zh_TW": "當探針中可靠點的比例低於此值時，該影格留空。用於防止曲線在樣本量悄悄塌陷時依然平滑。",
        "ja": "プローブの信頼できる点の割合がこの値を下回るフレームは空欄になります。標本数が減っていくのに曲線が滑らかなまま見えることを防ぎます。",
        "ko": "프로브의 신뢰할 수 있는 점 비율이 이 값보다 낮은 프레임은 비워 둡니다. 표본이 줄어드는데도 곡선이 매끄럽게 보이는 것을 막습니다.",
        "de": "Ein Bild bleibt leer, wenn weniger als dieser Anteil der Sondenpunkte zuverlässig ist. Verhindert eine Kurve, die glatt bleibt, während ihre Stichprobe verschwindet.",
        "fr": "Une image reste vide lorsque moins que cette fraction des points de la sonde est fiable. Évite une courbe qui reste lisse pendant que son échantillon disparaît.",
        "es": "Un fotograma se deja vacío cuando menos de esta fracción de los puntos de la sonda es fiable. Evita una curva que sigue siendo suave mientras su muestra se reduce.",
    },
    "Run a DIC analysis to plot probes.": {
        "zh_CN": "请先运行 DIC 分析，然后才能绘制探针曲线。",
        "zh_TW": "請先執行 DIC 分析，才能繪製探針曲線。",
        "ja": "プローブを描画するには、先に DIC 解析を実行してください。",
        "ko": "프로브를 그리려면 먼저 DIC 분석을 실행하세요.",
        "de": "Führen Sie eine DIC-Analyse aus, um Sonden darzustellen.",
        "fr": "Lancez une analyse DIC pour tracer les sondes.",
        "es": "Ejecute un análisis DIC para representar las sondas.",
    },
    "Place a probe on the reference image to begin.": {
        "zh_CN": "在参考图像上放置一个探针即可开始。",
        "zh_TW": "在參考影像上放置一個探針即可開始。",
        "ja": "参照画像にプローブを配置すると始まります。",
        "ko": "기준 이미지에 프로브를 배치하면 시작됩니다.",
        "de": "Setzen Sie eine Sonde auf das Referenzbild, um zu beginnen.",
        "fr": "Placez une sonde sur l'image de référence pour commencer.",
        "es": "Coloque una sonda en la imagen de referencia para empezar.",
    },
    "This statistic does not apply here.": {
        "zh_CN": "该统计量不适用于此处。",
        "zh_TW": "此統計量不適用於此處。",
        "ja": "この統計量はここでは適用できません。",
        "ko": "이 통계량은 여기에 적용되지 않습니다.",
        "de": "Diese Statistik ist hier nicht anwendbar.",
        "fr": "Cette statistique ne s'applique pas ici.",
        "es": "Este estadístico no se aplica aquí.",
    },
    "Shaded frames: %1": {
        "zh_CN": "阴影帧：%1",
        "zh_TW": "陰影影格：%1",
        "ja": "網掛けフレーム：%1",
        "ko": "음영 처리된 프레임: %1",
        "de": "Schattierte Bilder: %1",
        "fr": "Images grisées : %1",
        "es": "Fotogramas sombreados: %1",
    },
    "crosses a crack": {
        "zh_CN": "跨越裂纹", "zh_TW": "跨越裂縫", "ja": "き裂を横断",
        "ko": "균열을 가로지름", "de": "kreuzt einen Riss",
        "fr": "traverse une fissure", "es": "cruza una fisura",
    },
    "too few valid points": {
        "zh_CN": "有效点过少", "zh_TW": "有效點過少", "ja": "有効な点が少なすぎます",
        "ko": "유효한 점이 너무 적음", "de": "zu wenige gültige Punkte",
        "fr": "trop peu de points valides", "es": "muy pocos puntos válidos",
    },
    "no data": {
        "zh_CN": "无数据", "zh_TW": "無資料", "ja": "データなし",
        "ko": "데이터 없음", "de": "keine Daten", "fr": "aucune donnée",
        "es": "sin datos",
    },
    "Distance along line (%1)": {
        "zh_CN": "沿线距离（%1）",
        "zh_TW": "沿線距離（%1）",
        "ja": "線に沿った距離（%1）",
        "ko": "선을 따른 거리(%1)",
        "de": "Abstand entlang der Linie (%1)",
        "fr": "Distance le long de la ligne (%1)",
        "es": "Distancia a lo largo de la línea (%1)",
    },
    "Export CSV…": {
        "zh_CN": "导出 CSV…", "zh_TW": "匯出 CSV…", "ja": "CSV をエクスポート…",
        "ko": "CSV 내보내기…", "de": "CSV exportieren…",
        "fr": "Exporter en CSV…", "es": "Exportar CSV…",
    },
    "Export Chart…": {
        "zh_CN": "导出图表…", "zh_TW": "匯出圖表…", "ja": "グラフをエクスポート…",
        "ko": "차트 내보내기…", "de": "Diagramm exportieren…",
        "fr": "Exporter le graphique…", "es": "Exportar gráfico…",
    },
    "Export Chart": {
        "zh_CN": "导出图表", "zh_TW": "匯出圖表", "ja": "グラフをエクスポート",
        "ko": "차트 내보내기", "de": "Diagramm exportieren",
        "fr": "Exporter le graphique", "es": "Exportar gráfico",
    },
    "Export Probe Data": {
        "zh_CN": "导出探针数据", "zh_TW": "匯出探針資料",
        "ja": "プローブデータをエクスポート", "ko": "프로브 데이터 내보내기",
        "de": "Sondendaten exportieren", "fr": "Exporter les données de sonde",
        "es": "Exportar datos de sonda",
    },
    "There is nothing to export yet.": {
        "zh_CN": "目前没有可导出的内容。",
        "zh_TW": "目前沒有可匯出的內容。",
        "ja": "エクスポートできるものがまだありません。",
        "ko": "아직 내보낼 내용이 없습니다.",
        "de": "Es gibt noch nichts zu exportieren.",
        "fr": "Il n'y a encore rien à exporter.",
        "es": "Todavía no hay nada que exportar.",
    },
    "Probe data written to %1": {
        "zh_CN": "探针数据已写入 %1",
        "zh_TW": "探針資料已寫入 %1",
        "ja": "プローブデータを %1 に書き込みました",
        "ko": "프로브 데이터를 %1에 기록했습니다",
        "de": "Sondendaten nach %1 geschrieben",
        "fr": "Données de sonde écrites dans %1",
        "es": "Datos de sonda escritos en %1",
    },
    "Probe export failed: %1": {
        "zh_CN": "探针导出失败：%1",
        "zh_TW": "探針匯出失敗：%1",
        "ja": "プローブのエクスポートに失敗しました：%1",
        "ko": "프로브 내보내기 실패: %1",
        "de": "Sondenexport fehlgeschlagen: %1",
        "fr": "Échec de l'export de la sonde : %1",
        "es": "Error al exportar la sonda: %1",
    },
    "Chart written to %1": {
        "zh_CN": "图表已写入 %1",
        "zh_TW": "圖表已寫入 %1",
        "ja": "グラフを %1 に書き込みました",
        "ko": "차트를 %1에 기록했습니다",
        "de": "Diagramm nach %1 geschrieben",
        "fr": "Graphique écrit dans %1",
        "es": "Gráfico escrito en %1",
    },
    "Chart export failed: %1": {
        "zh_CN": "图表导出失败：%1",
        "zh_TW": "圖表匯出失敗：%1",
        "ja": "グラフのエクスポートに失敗しました：%1",
        "ko": "차트 내보내기 실패: %1",
        "de": "Diagrammexport fehlgeschlagen: %1",
        "fr": "Échec de l'export du graphique : %1",
        "es": "Error al exportar el gráfico: %1",
    },
    "CSV Files": {
        "zh_CN": "CSV 文件", "zh_TW": "CSV 檔案", "ja": "CSV ファイル",
        "ko": "CSV 파일", "de": "CSV-Dateien", "fr": "Fichiers CSV",
        "es": "Archivos CSV",
    },
    "PDF Documents": {
        "zh_CN": "PDF 文档", "zh_TW": "PDF 文件", "ja": "PDF ドキュメント",
        "ko": "PDF 문서", "de": "PDF-Dokumente", "fr": "Documents PDF",
        "es": "Documentos PDF",
    },
    "pyALDIC has hit an error": {
        "zh_CN": "pyALDIC 发生错误",
        "zh_TW": "pyALDIC 發生錯誤",
        "ja":    "pyALDIC でエラーが発生しました",
        "ko":    "pyALDIC에서 오류가 발생했습니다",
        "de":    "In pyALDIC ist ein Fehler aufgetreten",
        "fr":    "pyALDIC a rencontré une erreur",
        "es":    "pyALDIC ha encontrado un error",
    },
    "An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.": {
        "zh_CN": "发生了意外错误。应用程序之后的行为可能不正常，建议保存会话并重新启动。",
        "zh_TW": "發生了非預期的錯誤。應用程式之後的行為可能不正常，建議儲存會話並重新啟動。",
        "ja":    "予期しないエラーが発生しました。以降アプリケーションが正しく動作しない可能性があるため、セッションを保存して再起動することを推奨します。",
        "ko":    "예기치 않은 오류가 발생했습니다. 이후 애플리케이션이 정상적으로 동작하지 않을 수 있으므로 세션을 저장하고 다시 시작하는 것을 권장합니다.",
        "de":    "Ein unerwarteter Fehler ist aufgetreten. Die Anwendung verhält sich möglicherweise nicht mehr korrekt; es wird empfohlen, die Sitzung zu speichern und neu zu starten.",
        "fr":    "Une erreur inattendue s'est produite. L'application risque de ne plus fonctionner correctement ; il est recommandé d'enregistrer la session et de redémarrer.",
        "es":    "Se ha producido un error inesperado. Es posible que la aplicación no funcione correctamente a partir de ahora; se recomienda guardar la sesión y reiniciar.",
    },
    "Details were written to %1": {
        "zh_CN": "详细信息已写入 %1",
        "zh_TW": "詳細資訊已寫入 %1",
        "ja":    "詳細を %1 に書き込みました",
        "ko":    "자세한 내용을 %1에 기록했습니다",
        "de":    "Details wurden nach %1 geschrieben",
        "fr":    "Les détails ont été écrits dans %1",
        "es":    "Los detalles se han escrito en %1",
    },
    "Preparing compute kernels in the background. The first analysis on a new installation takes longer than the rest.": {
        "zh_CN": "正在后台准备计算内核。新安装后的首次分析会比之后的耗时更长。",
        "zh_TW": "正在背景準備計算核心。新安裝後的首次分析會比之後的耗時更長。",
        "ja":    "バックグラウンドで計算カーネルを準備しています。新規インストール後の最初の解析は、以降より時間がかかります。",
        "ko":    "백그라운드에서 계산 커널을 준비하고 있습니다. 새로 설치한 후 첫 번째 분석은 이후보다 오래 걸립니다.",
        "de":    "Rechenkernel werden im Hintergrund vorbereitet. Die erste Analyse einer neuen Installation dauert länger als die folgenden.",
        "fr":    "Préparation des noyaux de calcul en arrière-plan. La première analyse après une nouvelle installation prend plus de temps que les suivantes.",
        "es":    "Preparando los núcleos de cálculo en segundo plano. El primer análisis tras una instalación nueva tarda más que los siguientes.",
    },
    "Compute kernels ready (%1 s).": {
        "zh_CN": "计算内核已就绪（%1 秒）。",
        "zh_TW": "計算核心已就緒（%1 秒）。",
        "ja":    "計算カーネルの準備が完了しました（%1 秒）。",
        "ko":    "계산 커널 준비 완료(%1초).",
        "de":    "Rechenkernel bereit (%1 s).",
        "fr":    "Noyaux de calcul prêts (%1 s).",
        "es":    "Núcleos de cálculo listos (%1 s).",
    },
    "No animation was written. See the log for details.": {
        "zh_CN": "未写入任何动画。详情请查看日志。",
        "zh_TW": "未寫入任何動畫。詳情請查看日誌。",
        "ja":    "アニメーションは書き込まれませんでした。詳細はログを参照してください。",
        "ko":    "애니메이션이 기록되지 않았습니다. 자세한 내용은 로그를 확인하세요.",
        "de":    "Es wurde keine Animation geschrieben. Einzelheiten siehe Protokoll.",
        "fr":    "Aucune animation n'a été écrite. Consultez le journal pour plus de détails.",
        "es":    "No se ha escrito ninguna animación. Consulte el registro para más detalles.",
    },
    "Save Region of Interest Mask": {
        "zh_CN": "保存感兴趣区域掩模",
        "zh_TW": "儲存感興趣區域遮罩",
        "ja":    "関心領域マスクを保存",
        "ko":    "관심 영역 마스크 저장",
        "de":    "Region-of-Interest-Maske speichern",
        "fr":    "Enregistrer le masque de région d'intérêt",
        "es":    "Guardar máscara de región de interés",
    },
    "Import Mask Image": {
        "zh_CN": "导入掩模图像",
        "zh_TW": "匯入遮罩影像",
        "ja":    "マスク画像をインポート",
        "ko":    "마스크 이미지 가져오기",
        "de":    "Maskenbild importieren",
        "fr":    "Importer une image de masque",
        "es":    "Importar imagen de máscara",
    },
    "PNG Images": {
        "zh_CN": "PNG 图像",
        "zh_TW": "PNG 影像",
        "ja":    "PNG 画像",
        "ko":    "PNG 이미지",
        "de":    "PNG-Bilder",
        "fr":    "Images PNG",
        "es":    "Imágenes PNG",
    },
    # ===== Analysis chart: why a frame is marked ==========================
    "crack": {
        "zh_CN": "裂纹",
        "zh_TW": "裂紋",
        "ja":    "き裂",
        "ko":    "균열",
        "de":    "Riss",
        "fr":    "fissure",
        "es":    "grieta",
    },
    "gauge endpoint lost": {
        "zh_CN": "量规端点失效",
        "zh_TW": "量規端點失效",
        "ja":    "ゲージ端点が無効",
        "ko":    "게이지 끝점 상실",
        "de":    "Endpunkt der Messstrecke verloren",
        "fr":    "extrémité de jauge perdue",
        "es":    "extremo del calibre perdido",
    },
    "not computed": {
        "zh_CN": "未计算",
        "zh_TW": "未計算",
        "ja":    "未計算",
        "ko":    "계산되지 않음",
        "de":    "nicht berechnet",
        "fr":    "non calculé",
        "es":    "no calculado",
    },
    "unreliable (strain edge trim)": {
        "zh_CN": "不可靠（应变边缘裁剪）",
        "zh_TW": "不可靠（應變邊緣裁剪）",
        "ja":    "信頼できない（ひずみの端部トリミング）",
        "ko":    "신뢰할 수 없음(변형률 가장자리 트림)",
        "de":    "unzuverlässig (Dehnungs-Randbeschnitt)",
        "fr":    "non fiable (rognage des bords de la déformation)",
        "es":    "no fiable (recorte de bordes de la deformación)",
    },
    # ===== Analysis tab, reworked (P2) =====================================
    'Note': {
        'zh_CN': '说明',
        'zh_TW': '說明',
        'ja': '備考',
        'ko': '비고',
        'de': 'Hinweis',
        'fr': 'Remarque',
        'es': 'Nota',
    },
    'Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.': {
        'zh_CN': '尚未计算应变。请在“应变场”页中计算，或改为绘制位移。',
        'zh_TW': '尚未計算應變。請在「應變場」頁中計算，或改為繪製位移。',
        'ja': 'ひずみはまだ計算されていません。「ひずみ場」タブで計算するか、変位を表示してください。',
        'ko': "변형률이 아직 계산되지 않았습니다. '변형률장' 탭에서 계산하거나 변위를 표시하십시오.",
        'de': 'Die Dehnung wurde noch nicht berechnet. Berechnen Sie sie im Reiter „Dehnungsfeld“ oder zeigen Sie eine Verschiebung an.',
        'fr': "La déformation n'a pas encore été calculée. Calculez-la dans l'onglet « Champ de déformation » ou affichez un déplacement.",
        'es': 'La deformación aún no se ha calculado. Calcúlela en la pestaña «Campo de deformación» o represente un desplazamiento.',
    },
    'Gauge quantities need a line probe.': {
        'zh_CN': '量规类物理量需要线探针。',
        'zh_TW': '量規類物理量需要線探針。',
        'ja': 'ゲージ量には線プローブが必要です。',
        'ko': '게이지 양에는 선 프로브가 필요합니다.',
        'de': 'Messstreckengrößen benötigen eine Liniensonde.',
        'fr': 'Les grandeurs de jauge nécessitent une sonde linéaire.',
        'es': 'Las magnitudes de calibre requieren una sonda de línea.',
    },
    'No visible probe can show this quantity.': {
        'zh_CN': '没有可见的探针能显示此物理量。',
        'zh_TW': '沒有可見的探針能顯示此物理量。',
        'ja': 'この量を表示できる可視プローブがありません。',
        'ko': '이 양을 표시할 수 있는 보이는 프로브가 없습니다.',
        'de': 'Keine sichtbare Sonde kann diese Größe anzeigen.',
        'fr': 'Aucune sonde visible ne peut afficher cette grandeur.',
        'es': 'Ninguna sonda visible puede mostrar esta magnitud.',
    },
    'no valid data: %1': {
        'zh_CN': '无有效数据：%1',
        'zh_TW': '無有效資料：%1',
        'ja': '有効なデータなし：%1',
        'ko': '유효한 데이터 없음: %1',
        'de': 'keine gültigen Daten: %1',
        'fr': 'aucune donnée valide : %1',
        'es': 'sin datos válidos: %1',
    },
    'crack from frame %1': {
        'zh_CN': '第 %1 帧起出现裂纹',
        'zh_TW': '自第 %1 影格起出現裂紋',
        'ja': 'フレーム %1 からき裂',
        'ko': '%1 프레임부터 균열',
        'de': 'Riss ab Bild %1',
        'fr': "fissure dès l'image %1",
        'es': 'grieta desde el fotograma %1',
    },
    'endpoint lost from frame %1': {
        'zh_CN': '第 %1 帧起端点失效',
        'zh_TW': '自第 %1 影格起端點失效',
        'ja': 'フレーム %1 から端点が無効',
        'ko': '%1 프레임부터 끝점 상실',
        'de': 'Endpunkt ab Bild %1 verloren',
        'fr': "extrémité perdue dès l'image %1",
        'es': 'extremo perdido desde el fotograma %1',
    },
    'gaps: too few valid points': {
        'zh_CN': '有缺口：有效点过少',
        'zh_TW': '有缺口：有效點過少',
        'ja': '欠損あり：有効点が少なすぎる',
        'ko': '공백: 유효 점이 너무 적음',
        'de': 'Lücken: zu wenige gültige Punkte',
        'fr': 'lacunes : trop peu de points valides',
        'es': 'huecos: muy pocos puntos válidos',
    },
    'gaps: unreliable strain': {
        'zh_CN': '有缺口：应变不可靠',
        'zh_TW': '有缺口：應變不可靠',
        'ja': '欠損あり：ひずみが信頼できない',
        'ko': '공백: 변형률 신뢰 불가',
        'de': 'Lücken: unzuverlässige Dehnung',
        'fr': 'lacunes : déformation non fiable',
        'es': 'huecos: deformación no fiable',
    },
    'not plotted: gauges need a line': {
        'zh_CN': '未绘制：量规需要线探针',
        'zh_TW': '未繪製：量規需要線探針',
        'ja': '未表示：ゲージには線が必要',
        'ko': '표시 안 됨: 게이지에는 선이 필요',
        'de': 'nicht dargestellt: Messstrecken benötigen eine Linie',
        'fr': 'non tracée : les jauges nécessitent une ligne',
        'es': 'no representada: los calibres necesitan una línea',
    },
    'not plotted: one point has no spread or coverage': {
        'zh_CN': '未绘制：单个点没有离散度或覆盖率',
        'zh_TW': '未繪製：單一點沒有離散度或覆蓋率',
        'ja': '未表示：1 点にはばらつきも被覆率もない',
        'ko': '표시 안 됨: 한 점에는 분산이나 적용 범위가 없음',
        'de': 'nicht dargestellt: ein einzelner Punkt hat weder Streuung noch Abdeckung',
        'fr': "non tracée : un seul point n'a ni dispersion ni couverture",
        'es': 'no representada: un solo punto no tiene dispersión ni cobertura',
    },
    'not plotted': {
        'zh_CN': '未绘制',
        'zh_TW': '未繪製',
        'ja': '未表示',
        'ko': '표시 안 됨',
        'de': 'nicht dargestellt',
        'fr': 'non tracée',
        'es': 'no representada',
    },
    'Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.': {
        'zh_CN': '点击两次：起点和终点。线探针同时也是虚拟引伸计和裂纹张开量规。',
        'zh_TW': '點擊兩次：起點和終點。線探針同時也是虛擬引伸計和裂紋張開量規。',
        'ja': '2 回クリック：始点と終点。線は仮想伸び計およびき裂開口ゲージとしても使えます。',
        'ko': '두 번 클릭: 시작점과 끝점. 선은 가상 신율계이자 균열 개구 게이지이기도 합니다.',
        'de': 'Zweimal klicken: Anfang und Ende. Eine Linie ist auch ein virtueller Dehnungsaufnehmer und eine Rissöffnungs-Messstrecke.',
        'fr': "Cliquez deux fois : début et fin. Une ligne est aussi un extensomètre virtuel et une jauge d'ouverture de fissure.",
        'es': 'Haga clic dos veces: inicio y fin. Una línea es también un extensómetro virtual y un calibre de apertura de grieta.',
    },
    'Plot:': {
        'zh_CN': '绘制：',
        'zh_TW': '繪製：',
        'ja': '表示：',
        'ko': '표시:',
        'de': 'Darstellen:',
        'fr': 'Tracer :',
        'es': 'Representar:',
    },
    'X axis:': {
        'zh_CN': 'X 轴：',
        'zh_TW': 'X 軸：',
        'ja': 'X 軸：',
        'ko': 'X 축:',
        'de': 'X-Achse:',
        'fr': 'Axe X :',
        'es': 'Eje X:',
    },
    'Strain as:': {
        'zh_CN': '应变显示为：',
        'zh_TW': '應變顯示為：',
        'ja': 'ひずみの表示：',
        'ko': '변형률 표시:',
        'de': 'Dehnung als:',
        'fr': 'Déformation en :',
        'es': 'Deformación en:',
    },
    'Min. valid fraction:': {
        'zh_CN': '最小有效比例：',
        'zh_TW': '最小有效比例：',
        'ja': '最小有効割合：',
        'ko': '최소 유효 비율:',
        'de': 'Min. gültiger Anteil:',
        'fr': 'Fraction valide min. :',
        'es': 'Fracción válida mín.:',
    },
    "A frame is left blank when fewer than this fraction of a line's or region's points are reliable. Guards against a curve that stays smooth while its sample shrinks away.": {
        'zh_CN': '当线或区域中可靠点的比例低于此值时，该帧留空。避免样本逐渐缩小而曲线依然平滑的假象。',
        'zh_TW': '當線或區域中可靠點的比例低於此值時，該影格留空。避免樣本逐漸縮小而曲線依然平滑的假象。',
        'ja': '線または領域の信頼できる点の割合がこの値を下回るフレームは空白になります。サンプルが減っていても曲線が滑らかに見えてしまうことを防ぎます。',
        'ko': '선이나 영역에서 신뢰할 수 있는 점의 비율이 이 값보다 낮으면 해당 프레임은 비워 둡니다. 표본이 줄어드는데도 곡선이 매끄럽게 유지되는 것을 방지합니다.',
        'de': 'Ein Bild bleibt leer, wenn weniger als dieser Anteil der Punkte einer Linie oder Region zuverlässig ist. Verhindert eine Kurve, die glatt bleibt, während ihre Stichprobe schrumpft.',
        'fr': "Une image reste vide lorsque moins de cette fraction des points d'une ligne ou d'une région est fiable. Évite une courbe qui reste lisse alors que son échantillon se réduit.",
        'es': 'Un fotograma queda en blanco cuando menos de esta fracción de los puntos de una línea o región es fiable. Evita una curva que sigue suave mientras su muestra se reduce.',
    },
    'Probe data (CSV)…': {
        'zh_CN': '探针数据（CSV）…',
        'zh_TW': '探針資料（CSV）…',
        'ja': 'プローブデータ（CSV）…',
        'ko': '프로브 데이터(CSV)…',
        'de': 'Sondendaten (CSV)…',
        'fr': 'Données des sondes (CSV)…',
        'es': 'Datos de las sondas (CSV)…',
    },
    'Chart image…': {
        'zh_CN': '图表图片…',
        'zh_TW': '圖表圖片…',
        'ja': 'グラフ画像…',
        'ko': '차트 이미지…',
        'de': 'Diagrammbild…',
        'fr': 'Image du graphique…',
        'es': 'Imagen del gráfico…',
    },
    'Displacement U': {
        'zh_CN': '位移 U',
        'zh_TW': '位移 U',
        'ja': '変位 U',
        'ko': '변위 U',
        'de': 'Verschiebung U',
        'fr': 'Déplacement U',
        'es': 'Desplazamiento U',
    },
    'Displacement V': {
        'zh_CN': '位移 V',
        'zh_TW': '位移 V',
        'ja': '変位 V',
        'ko': '변위 V',
        'de': 'Verschiebung V',
        'fr': 'Déplacement V',
        'es': 'Desplazamiento V',
    },
    'Displacement magnitude': {
        'zh_CN': '位移大小',
        'zh_TW': '位移大小',
        'ja': '変位の大きさ',
        'ko': '변위 크기',
        'de': 'Verschiebungsbetrag',
        'fr': 'Norme du déplacement',
        'es': 'Magnitud del desplazamiento',
    },
    'Extensometer strain': {
        'zh_CN': '引伸计应变',
        'zh_TW': '引伸計應變',
        'ja': '伸び計ひずみ',
        'ko': '신율계 변형률',
        'de': 'Dehnungsaufnehmer-Dehnung',
        'fr': "Déformation d'extensomètre",
        'es': 'Deformación del extensómetro',
    },
    'Extensometer true strain': {
        'zh_CN': '引伸计真应变',
        'zh_TW': '引伸計真應變',
        'ja': '伸び計の真ひずみ',
        'ko': '신율계 진변형률',
        'de': 'Wahre Dehnung (Dehnungsaufnehmer)',
        'fr': "Déformation vraie d'extensomètre",
        'es': 'Deformación verdadera del extensómetro',
    },
    'Elongation ΔL': {
        'zh_CN': '伸长量 ΔL',
        'zh_TW': '伸長量 ΔL',
        'ja': '伸び ΔL',
        'ko': '신장량 ΔL',
        'de': 'Verlängerung ΔL',
        'fr': 'Allongement ΔL',
        'es': 'Alargamiento ΔL',
    },
    'Crack sliding': {
        'zh_CN': '裂纹滑移',
        'zh_TW': '裂紋滑移',
        'ja': 'き裂すべり',
        'ko': '균열 미끄럼',
        'de': 'Rissgleiten',
        'fr': 'Glissement de fissure',
        'es': 'Deslizamiento de grieta',
    },
    'Crack opening magnitude': {
        'zh_CN': '裂纹张开位移大小',
        'zh_TW': '裂紋張開位移大小',
        'ja': 'き裂開口変位の大きさ',
        'ko': '균열 개구 변위 크기',
        'de': 'Betrag der Rissöffnung',
        'fr': "Norme de l'ouverture de fissure",
        'es': 'Magnitud de la apertura de grieta',
    },
    'Time (s)': {
        'zh_CN': '时间 (s)',
        'zh_TW': '時間 (s)',
        'ja': '時間 (s)',
        'ko': '시간 (s)',
        'de': 'Zeit (s)',
        'fr': 'Temps (s)',
        'es': 'Tiempo (s)',
    },
    'ratio': {
        'zh_CN': '比值',
        'zh_TW': '比值',
        'ja': '比率',
        'ko': '비율',
        'de': 'Verhältnis',
        'fr': 'rapport',
        'es': 'proporción',
    },
    # ===== Analysis canvas (P3) ============================================
    'Virtual extensometer': {
        'zh_CN': '虚拟引伸计',
        'zh_TW': '虛擬引伸計',
        'ja': '仮想伸び計',
        'ko': '가상 신율계',
        'de': 'Virtueller Dehnungsaufnehmer',
        'fr': 'Extensomètre virtuel',
        'es': 'Extensómetro virtual',
    },
    'Crack gauge': {
        'zh_CN': '裂纹量规',
        'zh_TW': '裂紋量規',
        'ja': 'き裂ゲージ',
        'ko': '균열 게이지',
        'de': 'Riss-Messstrecke',
        'fr': 'Jauge de fissure',
        'es': 'Calibre de grieta',
    },
    'Click the two gauge points. The chart then shows the strain between them.': {
        'zh_CN': '点击两个标距点。图表随后显示两点之间的应变。',
        'zh_TW': '點擊兩個標距點。圖表隨後顯示兩點之間的應變。',
        'ja': '2 つの標点をクリックします。グラフにはその間のひずみが表示されます。',
        'ko': '두 개의 표점을 클릭하십시오. 차트에 두 점 사이의 변형률이 표시됩니다.',
        'de': 'Klicken Sie auf die beiden Messpunkte. Das Diagramm zeigt dann die Dehnung zwischen ihnen.',
        'fr': 'Cliquez sur les deux points de mesure. Le graphique affiche alors la déformation entre eux.',
        'es': 'Haga clic en los dos puntos de medida. El gráfico muestra entonces la deformación entre ellos.',
    },
    'Click one point on each side of the crack. The chart then shows how far it opens.': {
        'zh_CN': '在裂纹两侧各点击一个点。图表随后显示裂纹的张开量。',
        'zh_TW': '在裂紋兩側各點擊一個點。圖表隨後顯示裂紋的張開量。',
        'ja': 'き裂の両側に 1 点ずつクリックします。グラフにはき裂の開口量が表示されます。',
        'ko': '균열 양쪽에 한 점씩 클릭하십시오. 차트에 균열이 벌어진 정도가 표시됩니다.',
        'de': 'Klicken Sie auf je einen Punkt auf beiden Seiten des Risses. Das Diagramm zeigt dann, wie weit er sich öffnet.',
        'fr': 'Cliquez sur un point de chaque côté de la fissure. Le graphique affiche alors son ouverture.',
        'es': 'Haga clic en un punto a cada lado de la grieta. El gráfico muestra entonces cuánto se abre.',
    },
    'Fit': {
        'zh_CN': '适配',
        'zh_TW': '適配',
        'ja': 'フィット',
        'ko': '맞춤',
        'de': 'Anpassen',
        'fr': 'Ajuster',
        'es': 'Ajustar',
    },
    'Fit image to viewport': {
        'zh_CN': '将图像适配到视口',
        'zh_TW': '將影像適配到視口',
        'ja': '画像をビューポートに合わせる',
        'ko': '이미지를 뷰포트에 맞춤',
        'de': 'Bild an den Ansichtsbereich anpassen',
        'fr': "Ajuster l'image à la vue",
        'es': 'Ajustar la imagen a la vista',
    },
    '100%': {
        'zh_CN': '100%',
        'zh_TW': '100%',
        'ja': '100%',
        'ko': '100%',
        'de': '100%',
        'fr': '100%',
        'es': '100%',
    },
    'Zoom to 100% (1:1)': {
        'zh_CN': '缩放到 100%（1:1）',
        'zh_TW': '縮放到 100%（1:1）',
        'ja': '100% (1:1) ズーム',
        'ko': '100%(1:1) 확대',
        'de': 'Auf 100% (1:1) zoomen',
        'fr': 'Zoomer à 100% (1:1)',
        'es': 'Zoom al 100% (1:1)',
    },
    'Zoom in': {
        'zh_CN': '放大',
        'zh_TW': '放大',
        'ja': '拡大',
        'ko': '확대',
        'de': 'Vergrößern',
        'fr': 'Zoom avant',
        'es': 'Acercar',
    },
    'Zoom out': {
        'zh_CN': '缩小',
        'zh_TW': '縮小',
        'ja': '縮小',
        'ko': '축소',
        'de': 'Verkleinern',
        'fr': 'Zoom arrière',
        'es': 'Alejar',
    },
    'Show field': {
        'zh_CN': '显示场',
        'zh_TW': '顯示場',
        'ja': '場を表示',
        'ko': '필드 표시',
        'de': 'Feld anzeigen',
        'fr': 'Afficher le champ',
        'es': 'Mostrar campo',
    },
    "Colour the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab's field is shown.": {
        'zh_CN': '用当前帧所绘物理量的场为参考图像着色。对于量规读数，显示“应变场”页中的场。',
        'zh_TW': '以目前影格所繪物理量的場為參考影像著色。對於量規讀數，顯示「應變場」頁中的場。',
        'ja': '現在のフレームで表示中の量の場で参照画像を色付けします。ゲージの読み取り値では「ひずみ場」タブの場を表示します。',
        'ko': "현재 프레임에서 표시 중인 양의 필드로 기준 이미지를 색칠합니다. 게이지 판독값의 경우 '변형률장' 탭의 필드를 표시합니다.",
        'de': 'Färbt das Referenzbild mit dem dargestellten Feld im aktuellen Bild ein. Bei einer Messstreckengröße wird das Feld des Reiters „Dehnungsfeld“ angezeigt.',
        'fr': "Colore l'image de référence avec le champ tracé à l'image courante. Pour une lecture de jauge, le champ de l'onglet « Champ de déformation » est affiché.",
        'es': 'Colorea la imagen de referencia con el campo representado en el fotograma actual. Para una lectura de calibre, se muestra el campo de la pestaña «Campo de deformación».',
    },
    'Could not draw the field: %1': {
        'zh_CN': '无法绘制场：%1',
        'zh_TW': '無法繪製場：%1',
        'ja': '場を描画できませんでした：%1',
        'ko': '필드를 그릴 수 없습니다: %1',
        'de': 'Das Feld konnte nicht gezeichnet werden: %1',
        'fr': 'Impossible de dessiner le champ : %1',
        'es': 'No se pudo dibujar el campo: %1',
    },
    'Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.': {
        'zh_CN': '拖动可移动探针，拖动控制点可改变其形状。按 Delete 删除，按 F2 重命名。',
        'zh_TW': '拖曳可移動探針，拖曳控制點可改變其形狀。按 Delete 刪除，按 F2 重新命名。',
        'ja': 'ドラッグでプローブを移動、ハンドルのドラッグで形状を変更します。Delete で削除、F2 で名前を変更します。',
        'ko': '드래그하여 프로브를 이동하거나 핸들을 드래그하여 모양을 바꿉니다. Delete로 삭제하고 F2로 이름을 바꿉니다.',
        'de': 'Ziehen verschiebt die Sonde, Ziehen an einem Griff ändert ihre Form. Entf löscht sie, F2 benennt sie um.',
        'fr': 'Faites glisser pour déplacer la sonde, ou une poignée pour la remodeler. Suppr la supprime ; F2 la renomme.',
        'es': 'Arrastre para mover la sonda, o un tirador para cambiar su forma. Supr la elimina; F2 le cambia el nombre.',
    },
    'not plotted: off the measured area': {
        'zh_CN': '未绘制：不在测量区域内',
        'zh_TW': '未繪製：不在量測區域內',
        'ja': '未表示：測定領域の外',
        'ko': '표시 안 됨: 측정 영역 밖',
        'de': 'nicht dargestellt: außerhalb des Messbereichs',
        'fr': 'non tracée : hors de la zone mesurée',
        'es': 'no representada: fuera del área medida',
    },
    'not plotted: a gauge end is off the measured area': {
        'zh_CN': '未绘制：量规端点不在测量区域内',
        'zh_TW': '未繪製：量規端點不在量測區域內',
        'ja': '未表示：ゲージの端点が測定領域の外',
        'ko': '표시 안 됨: 게이지 끝점이 측정 영역 밖',
        'de': 'nicht dargestellt: ein Messstreckenende liegt außerhalb des Messbereichs',
        'fr': 'non tracée : une extrémité de la jauge est hors de la zone mesurée',
        'es': 'no representada: un extremo del calibre está fuera del área medida',
    },
    # ===== Analysis line views (P4) =========================================
    'Over time': {
        'zh_CN': '随时间变化',
        'zh_TW': '隨時間變化',
        'ja': '時間変化',
        'ko': '시간 변화',
        'de': 'Zeitverlauf',
        'fr': 'Évolution temporelle',
        'es': 'Evolución temporal',
    },
    "Each probe's reading at every frame.": {
        'zh_CN': '每个探针在各帧的读数。',
        'zh_TW': '每個探針在各影格的讀數。',
        'ja': '各プローブの全フレームでの読み取り値。',
        'ko': '각 프로브의 모든 프레임에서의 판독값.',
        'de': 'Der Messwert jeder Sonde in jedem Bild.',
        'fr': 'La lecture de chaque sonde à chaque image.',
        'es': 'La lectura de cada sonda en cada fotograma.',
    },
    'Along the line': {
        'zh_CN': '沿线分布',
        'zh_TW': '沿線分佈',
        'ja': '線に沿った分布',
        'ko': '선을 따른 분포',
        'de': 'Entlang der Linie',
        'fr': 'Le long de la ligne',
        'es': 'A lo largo de la línea',
    },
    'The field along the selected line at the current frame, over the other frames in grey.': {
        'zh_CN': '当前帧沿所选线的场分布，其它帧以灰色显示在下方。',
        'zh_TW': '目前影格沿所選線的場分佈，其他影格以灰色顯示在下方。',
        'ja': '現在のフレームにおける選択した線に沿った場。他のフレームは背後に灰色で表示されます。',
        'ko': '현재 프레임에서 선택한 선을 따른 필드이며, 다른 프레임은 뒤에 회색으로 표시됩니다.',
        'de': 'Das Feld entlang der ausgewählten Linie im aktuellen Bild, dahinter die übrigen Bilder in Grau.',
        'fr': "Le champ le long de la ligne sélectionnée à l'image courante, sur les autres images en gris.",
        'es': 'El campo a lo largo de la línea seleccionada en el fotograma actual, sobre los demás fotogramas en gris.',
    },
    'Kymograph': {
        'zh_CN': '时空图',
        'zh_TW': '時空圖',
        'ja': 'キモグラフ',
        'ko': '키모그래프',
        'de': 'Kymogramm',
        'fr': 'Kymographe',
        'es': 'Quimograma',
    },
    'The field along the selected line at every frame: distance against frame, value as colour.': {
        'zh_CN': '所选线在每一帧的场分布：距离对帧，数值以颜色表示。',
        'zh_TW': '所選線在每一影格的場分佈：距離對影格，數值以顏色表示。',
        'ja': '選択した線に沿った全フレームの場：距離とフレームの関係を、値を色で表示します。',
        'ko': '모든 프레임에서 선택한 선을 따른 필드: 거리 대 프레임, 값은 색으로 표시.',
        'de': 'Das Feld entlang der ausgewählten Linie in jedem Bild: Abstand über Bild, Wert als Farbe.',
        'fr': "Le champ le long de la ligne sélectionnée à chaque image : distance en fonction de l'image, valeur en couleur.",
        'es': 'El campo a lo largo de la línea seleccionada en cada fotograma: distancia frente a fotograma, valor como color.',
    },
    'Other frames': {
        'zh_CN': '其它帧',
        'zh_TW': '其他影格',
        'ja': '他のフレーム',
        'ko': '다른 프레임',
        'de': 'Übrige Bilder',
        'fr': 'Autres images',
        'es': 'Otros fotogramas',
    },
    "Draw the other frames' profiles faintly behind the current one (at most twelve, evenly spaced).": {
        'zh_CN': '在当前帧后方淡色绘制其它帧的分布（最多十二帧，均匀间隔）。',
        'zh_TW': '在目前影格後方以淡色繪製其他影格的分佈（最多十二個，均勻間隔）。',
        'ja': '現在のフレームの背後に他のフレームの分布を薄く描画します（最大 12 フレーム、等間隔）。',
        'ko': '현재 프레임 뒤에 다른 프레임의 분포를 흐리게 그립니다(최대 12개, 균등 간격).',
        'de': 'Die Profile der übrigen Bilder blass hinter dem aktuellen zeichnen (höchstens zwölf, gleichmäßig verteilt).',
        'fr': "Tracer en pâle les profils des autres images derrière l'image courante (au plus douze, régulièrement espacées).",
        'es': 'Dibujar tenuemente los perfiles de los demás fotogramas detrás del actual (como máximo doce, espaciados uniformemente).',
    },
    'Line data (CSV)…': {
        'zh_CN': '线数据（CSV）…',
        'zh_TW': '線資料（CSV）…',
        'ja': '線データ（CSV）…',
        'ko': '선 데이터(CSV)…',
        'de': 'Liniendaten (CSV)…',
        'fr': 'Données de ligne (CSV)…',
        'es': 'Datos de línea (CSV)…',
    },
    'A line view shows a field. Choose a field to plot.': {
        'zh_CN': '沿线视图显示的是场。请选择一个场来绘制。',
        'zh_TW': '沿線視圖顯示的是場。請選擇一個場來繪製。',
        'ja': '線のビューは場を表示します。表示する場を選んでください。',
        'ko': '선 보기는 필드를 표시합니다. 표시할 필드를 선택하십시오.',
        'de': 'Eine Linienansicht zeigt ein Feld. Wählen Sie ein Feld zum Darstellen.',
        'fr': 'Une vue de ligne affiche un champ. Choisissez un champ à tracer.',
        'es': 'Una vista de línea muestra un campo. Elija un campo para representar.',
    },
    'Place a line probe, or select one, to see the field along it.': {
        'zh_CN': '放置或选择一个线探针，即可查看沿线的场。',
        'zh_TW': '放置或選擇一個線探針，即可查看沿線的場。',
        'ja': '線プローブを配置または選択すると、線に沿った場が表示されます。',
        'ko': '선 프로브를 배치하거나 선택하면 선을 따른 필드를 볼 수 있습니다.',
        'de': 'Setzen oder wählen Sie eine Liniensonde, um das Feld entlang der Linie zu sehen.',
        'fr': 'Placez ou sélectionnez une sonde linéaire pour voir le champ le long de celle-ci.',
        'es': 'Coloque o seleccione una sonda de línea para ver el campo a lo largo de ella.',
    },
    'Distance along %1 (%2)': {
        'zh_CN': '沿 %1 的距离（%2）',
        'zh_TW': '沿 %1 的距離（%2）',
        'ja': '%1 に沿った距離（%2）',
        'ko': '%1을(를) 따른 거리(%2)',
        'de': 'Abstand entlang %1 (%2)',
        'fr': 'Distance le long de %1 (%2)',
        'es': 'Distancia a lo largo de %1 (%2)',
    },
    '%1, frame %2': {
        'zh_CN': '%1，第 %2 帧',
        'zh_TW': '%1，第 %2 影格',
        'ja': '%1、フレーム %2',
        'ko': '%1, %2 프레임',
        'de': '%1, Bild %2',
        'fr': '%1, image %2',
        'es': '%1, fotograma %2',
    },
    'Export Line Data': {
        'zh_CN': '导出线数据',
        'zh_TW': '匯出線資料',
        'ja': '線データをエクスポート',
        'ko': '선 데이터 내보내기',
        'de': 'Liniendaten exportieren',
        'fr': 'Exporter les données de ligne',
        'es': 'Exportar datos de línea',
    },
    'Line export failed: %1': {
        'zh_CN': '线数据导出失败：%1',
        'zh_TW': '線資料匯出失敗：%1',
        'ja': '線データのエクスポートに失敗しました：%1',
        'ko': '선 데이터 내보내기 실패: %1',
        'de': 'Linienexport fehlgeschlagen: %1',
        'fr': "Échec de l'export de la ligne : %1",
        'es': 'Error al exportar la línea: %1',
    },
    'Nothing valid along %1: its strain is trimmed as low-confidence near an edge or a hole. Plot a displacement, or trim less on the Strain Field tab.': {
        'zh_CN': '%1 沿线没有有效数据：其应变在边缘或孔附近被判为低置信度而裁剪。请改为绘制位移，或在“应变场”页减少裁剪。',
        'zh_TW': '%1 沿線沒有有效資料：其應變在邊緣或孔附近被判為低可信度而裁剪。請改為繪製位移，或在「應變場」頁減少裁剪。',
        'ja': '%1 に沿って有効なデータがありません：縁や穴の近くのひずみは低信頼度として除去されています。変位を表示するか、「ひずみ場」タブで除去を減らしてください。',
        'ko': "%1을(를) 따라 유효한 데이터가 없습니다: 가장자리나 구멍 근처의 변형률이 저신뢰도로 잘려 나갔습니다. 변위를 표시하거나 '변형률장' 탭에서 잘라내기를 줄이십시오.",
        'de': 'Keine gültigen Daten entlang %1: Die Dehnung wird nahe einem Rand oder Loch als wenig zuverlässig beschnitten. Stellen Sie eine Verschiebung dar oder beschneiden Sie im Reiter „Dehnungsfeld“ weniger.',
        'fr': "Aucune donnée valide le long de %1 : sa déformation est rognée comme peu fiable près d'un bord ou d'un trou. Tracez un déplacement, ou rognez moins dans l'onglet « Champ de déformation ».",
        'es': 'No hay datos válidos a lo largo de %1: su deformación se recorta como poco fiable cerca de un borde o un agujero. Represente un desplazamiento, o recorte menos en la pestaña «Campo de deformación».',
    },
    'Nothing valid along %1: a crack has consumed the material under it.': {
        'zh_CN': '%1 沿线没有有效数据：其下方的材料已被裂纹消耗。',
        'zh_TW': '%1 沿線沒有有效資料：其下方的材料已被裂紋消耗。',
        'ja': '%1 に沿って有効なデータがありません：下の材料がき裂によって失われました。',
        'ko': '%1을(를) 따라 유효한 데이터가 없습니다: 아래의 재료가 균열로 소실되었습니다.',
        'de': 'Keine gültigen Daten entlang %1: Ein Riss hat das Material darunter aufgezehrt.',
        'fr': 'Aucune donnée valide le long de %1 : une fissure a consommé la matière en dessous.',
        'es': 'No hay datos válidos a lo largo de %1: una grieta ha consumido el material que hay debajo.',
    },
    'Nothing valid along %1: it lies off the measured area.': {
        'zh_CN': '%1 沿线没有有效数据：它不在测量区域内。',
        'zh_TW': '%1 沿線沒有有效資料：它不在量測區域內。',
        'ja': '%1 に沿って有効なデータがありません：測定領域の外にあります。',
        'ko': '%1을(를) 따라 유효한 데이터가 없습니다: 측정 영역 밖에 있습니다.',
        'de': 'Keine gültigen Daten entlang %1: Die Linie liegt außerhalb des Messbereichs.',
        'fr': 'Aucune donnée valide le long de %1 : elle est hors de la zone mesurée.',
        'es': 'No hay datos válidos a lo largo de %1: está fuera del área medida.',
    },
    'Line data written to %1': {
        'zh_CN': '线数据已写入 %1',
        'zh_TW': '線資料已寫入 %1',
        'ja': '線データを %1 に書き込みました',
        'ko': '선 데이터를 %1에 기록했습니다',
        'de': 'Liniendaten nach %1 geschrieben',
        'fr': 'Données de ligne écrites dans %1',
        'es': 'Datos de línea escritos en %1',
    },
    # ===== Analysis publication output (P5) ================================
    'Copy chart': {
        'zh_CN': '复制图表',
        'zh_TW': '複製圖表',
        'ja': 'グラフをコピー',
        'ko': '차트 복사',
        'de': 'Diagramm kopieren',
        'fr': 'Copier le graphique',
        'es': 'Copiar el gráfico',
    },
    'Copy plotted data': {
        'zh_CN': '复制所绘数据',
        'zh_TW': '複製所繪資料',
        'ja': '表示中のデータをコピー',
        'ko': '표시된 데이터 복사',
        'de': 'Dargestellte Daten kopieren',
        'fr': 'Copier les données tracées',
        'es': 'Copiar los datos representados',
    },
    'SVG Images': {
        'zh_CN': 'SVG 图像',
        'zh_TW': 'SVG 影像',
        'ja': 'SVG 画像',
        'ko': 'SVG 이미지',
        'de': 'SVG-Bilder',
        'fr': 'Images SVG',
        'es': 'Imágenes SVG',
    },
    'Chart copied to the clipboard.': {
        'zh_CN': '图表已复制到剪贴板。',
        'zh_TW': '圖表已複製到剪貼簿。',
        'ja': 'グラフをクリップボードにコピーしました。',
        'ko': '차트를 클립보드에 복사했습니다.',
        'de': 'Diagramm in die Zwischenablage kopiert.',
        'fr': 'Graphique copié dans le presse-papiers.',
        'es': 'Gráfico copiado al portapapeles.',
    },
    'Plotted data copied to the clipboard.': {
        'zh_CN': '所绘数据已复制到剪贴板。',
        'zh_TW': '所繪資料已複製到剪貼簿。',
        'ja': '表示中のデータをクリップボードにコピーしました。',
        'ko': '표시된 데이터를 클립보드에 복사했습니다.',
        'de': 'Dargestellte Daten in die Zwischenablage kopiert.',
        'fr': 'Données tracées copiées dans le presse-papiers.',
        'es': 'Datos representados copiados al portapapeles.',
    },
    # ===== Analysis load data (P6b) =======================================
    'Load data…': {
        'zh_CN': '载荷数据…',
        'zh_TW': '載荷資料…',
        'ja': '荷重データ…',
        'ko': '하중 데이터…',
        'de': 'Kraftdaten…',
        'fr': 'Données de charge…',
        'es': 'Datos de carga…',
    },
    "Import a testing machine's load record (CSV) to plot against load or stress, and to draw stress-strain curves.": {
        'zh_CN': '导入试验机的载荷记录（CSV），以载荷或应力为横轴绘图，并绘制应力–应变曲线。',
        'zh_TW': '匯入試驗機的載荷記錄（CSV），以載荷或應力為橫軸繪圖，並繪製應力–應變曲線。',
        'ja': '試験機の荷重記録（CSV）を読み込み、荷重または応力に対してプロットし、応力–ひずみ曲線を描きます。',
        'ko': '시험기의 하중 기록(CSV)을 가져와 하중 또는 응력에 대해 그래프를 그리고 응력–변형률 곡선을 그립니다.',
        'de': 'Kraftaufzeichnung einer Prüfmaschine (CSV) importieren, um über Kraft oder Spannung darzustellen und Spannungs-Dehnungs-Kurven zu zeichnen.',
        'fr': "Importer l'enregistrement de charge d'une machine d'essai (CSV) pour tracer en fonction de la charge ou de la contrainte, et tracer des courbes contrainte-déformation.",
        'es': 'Importar el registro de carga de una máquina de ensayo (CSV) para representar frente a la carga o la tensión y trazar curvas tensión-deformación.',
    },
    'Load (N)': {
        'zh_CN': '载荷 (N)',
        'zh_TW': '載荷 (N)',
        'ja': '荷重 (N)',
        'ko': '하중 (N)',
        'de': 'Kraft (N)',
        'fr': 'Charge (N)',
        'es': 'Carga (N)',
    },
    'Stress (MPa)': {
        'zh_CN': '应力 (MPa)',
        'zh_TW': '應力 (MPa)',
        'ja': '応力 (MPa)',
        'ko': '응력 (MPa)',
        'de': 'Spannung (MPa)',
        'fr': 'Contrainte (MPa)',
        'es': 'Tensión (MPa)',
    },
    'The load data cannot be matched to the frames: %1': {
        'zh_CN': '载荷数据无法与帧对应：%1',
        'zh_TW': '載荷資料無法與影格對應：%1',
        'ja': '荷重データをフレームに対応付けられません：%1',
        'ko': '하중 데이터를 프레임에 맞출 수 없습니다: %1',
        'de': 'Die Kraftdaten lassen sich den Bildern nicht zuordnen: %1',
        'fr': 'Les données de charge ne peuvent pas être associées aux images : %1',
        'es': 'Los datos de carga no se pueden asociar a los fotogramas: %1',
    },
    'Load Data': {
        'zh_CN': '载荷数据',
        'zh_TW': '載荷資料',
        'ja': '荷重データ',
        'ko': '하중 데이터',
        'de': 'Kraftdaten',
        'fr': 'Données de charge',
        'es': 'Datos de carga',
    },
    'No file chosen.': {
        'zh_CN': '未选择文件。',
        'zh_TW': '未選擇檔案。',
        'ja': 'ファイルが選択されていません。',
        'ko': '선택한 파일이 없습니다.',
        'de': 'Keine Datei gewählt.',
        'fr': 'Aucun fichier choisi.',
        'es': 'Ningún archivo elegido.',
    },
    'Choose file…': {
        'zh_CN': '选择文件…',
        'zh_TW': '選擇檔案…',
        'ja': 'ファイルを選択…',
        'ko': '파일 선택…',
        'de': 'Datei wählen…',
        'fr': 'Choisir un fichier…',
        'es': 'Elegir archivo…',
    },
    'Load column:': {
        'zh_CN': '载荷列：',
        'zh_TW': '載荷欄：',
        'ja': '荷重の列：',
        'ko': '하중 열:',
        'de': 'Kraftspalte:',
        'fr': 'Colonne de charge :',
        'es': 'Columna de carga:',
    },
    'By time': {
        'zh_CN': '按时间',
        'zh_TW': '按時間',
        'ja': '時間で',
        'ko': '시간으로',
        'de': 'Nach Zeit',
        'fr': 'Par le temps',
        'es': 'Por tiempo',
    },
    'By frame number': {
        'zh_CN': '按帧号',
        'zh_TW': '按影格編號',
        'ja': 'フレーム番号で',
        'ko': '프레임 번호로',
        'de': 'Nach Bildnummer',
        'fr': "Par numéro d'image",
        'es': 'Por número de fotograma',
    },
    'Match rows to frames:': {
        'zh_CN': '行与帧的对应方式：',
        'zh_TW': '列與影格的對應方式：',
        'ja': '行とフレームの対応：',
        'ko': '행과 프레임 연결:',
        'de': 'Zeilen den Bildern zuordnen:',
        'fr': 'Associer les lignes aux images :',
        'es': 'Asociar filas a fotogramas:',
    },
    'Time column:': {
        'zh_CN': '时间列：',
        'zh_TW': '時間欄：',
        'ja': '時間の列：',
        'ko': '시간 열:',
        'de': 'Zeitspalte:',
        'fr': 'Colonne de temps :',
        'es': 'Columna de tiempo:',
    },
    'Offset:': {
        'zh_CN': '偏移：',
        'zh_TW': '偏移：',
        'ja': 'オフセット：',
        'ko': '오프셋:',
        'de': 'Versatz:',
        'fr': 'Décalage :',
        'es': 'Desfase:',
    },
    "The machine's time at the reference image. If the camera started 2 s after the machine, enter 2.": {
        'zh_CN': '参考图像时刻对应的试验机时间。若相机比试验机晚 2 s 启动，则输入 2。',
        'zh_TW': '參考影像時刻對應的試驗機時間。若相機比試驗機晚 2 s 啟動，則輸入 2。',
        'ja': '参照画像の時点における試験機の時間。カメラが試験機より 2 s 遅れて開始した場合は 2 と入力します。',
        'ko': '기준 이미지 시점의 시험기 시간입니다. 카메라가 시험기보다 2 s 늦게 시작했다면 2를 입력하십시오.',
        'de': 'Die Maschinenzeit beim Referenzbild. Startete die Kamera 2 s nach der Maschine, geben Sie 2 ein.',
        'fr': "Le temps de la machine à l'image de référence. Si la caméra a démarré 2 s après la machine, saisissez 2.",
        'es': 'El tiempo de la máquina en la imagen de referencia. Si la cámara empezó 2 s después que la máquina, introduzca 2.',
    },
    'Frame column:': {
        'zh_CN': '帧号列：',
        'zh_TW': '影格欄：',
        'ja': 'フレームの列：',
        'ko': '프레임 열:',
        'de': 'Bildspalte:',
        'fr': "Colonne d'image :",
        'es': 'Columna de fotograma:',
    },
    'The first image is numbered:': {
        'zh_CN': '第一张图像的编号：',
        'zh_TW': '第一張影像的編號：',
        'ja': '最初の画像の番号：',
        'ko': '첫 이미지의 번호:',
        'de': 'Das erste Bild hat die Nummer:',
        'fr': 'Numéro de la première image :',
        'es': 'Número de la primera imagen:',
    },
    'Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.': {
        'zh_CN': '初始横截面积，用于计算工程应力 F / A0（MPa）。仅需载荷时保持为 0。',
        'zh_TW': '初始橫截面積，用於計算工程應力 F / A0（MPa）。僅需載荷時保持為 0。',
        'ja': '初期断面積。公称応力 F / A0（MPa）の計算に使います。荷重のみの場合は 0 のままにします。',
        'ko': '초기 단면적으로, 공칭 응력 F / A0(MPa) 계산에 사용합니다. 하중만 필요하면 0으로 두십시오.',
        'de': 'Anfangsquerschnitt für die technische Spannung F / A0 in MPa. Für nur Kraft bei 0 lassen.',
        'fr': 'Section initiale, pour la contrainte nominale F / A0 en MPa. Laisser à 0 pour la charge seule.',
        'es': 'Sección inicial, para la tensión ingenieril F / A0 en MPa. Déjela en 0 para usar solo la carga.',
    },
    'Cross-section A0:': {
        'zh_CN': '横截面积 A0：',
        'zh_TW': '橫截面積 A0：',
        'ja': '断面積 A0：',
        'ko': '단면적 A0:',
        'de': 'Querschnitt A0:',
        'fr': 'Section A0 :',
        'es': 'Sección A0:',
    },
    'Remove Load Data': {
        'zh_CN': '移除载荷数据',
        'zh_TW': '移除載荷資料',
        'ja': '荷重データを削除',
        'ko': '하중 데이터 제거',
        'de': 'Kraftdaten entfernen',
        'fr': 'Supprimer les données de charge',
        'es': 'Quitar los datos de carga',
    },
    'Camera frame rate: %1 fps, from Physical Units.': {
        'zh_CN': '相机帧率：%1 fps，取自“物理单位”。',
        'zh_TW': '相機影格率：%1 fps，取自「物理單位」。',
        'ja': 'カメラのフレームレート：%1 fps（「物理単位」の設定）。',
        'ko': "카메라 프레임 속도: %1 fps, '물리 단위'에서 설정.",
        'de': 'Kamera-Bildrate: %1 fps, aus „Physikalische Einheiten“.',
        'fr': "Fréquence d'images de la caméra : %1 fps, issue des « Unités physiques ».",
        'es': 'Velocidad de fotogramas de la cámara: %1 fps, de «Unidades físicas».',
    },
    'Set the camera frame rate under Physical Units to match by time.': {
        'zh_CN': '请在“物理单位”中设置相机帧率，才能按时间对应。',
        'zh_TW': '請在「物理單位」中設定相機影格率，才能按時間對應。',
        'ja': '時間で対応付けるには、「物理単位」でカメラのフレームレートを設定してください。',
        'ko': "시간으로 맞추려면 '물리 단위'에서 카메라 프레임 속도를 설정하십시오.",
        'de': 'Legen Sie unter „Physikalische Einheiten“ die Kamera-Bildrate fest, um nach Zeit zuzuordnen.',
        'fr': "Définissez la fréquence d'images de la caméra dans « Unités physiques » pour associer par le temps.",
        'es': 'Defina la velocidad de fotogramas de la cámara en «Unidades físicas» para asociar por tiempo.',
    },
    '(unnamed)': {
        'zh_CN': '（未命名）',
        'zh_TW': '（未命名）',
        'ja': '（名前なし）',
        'ko': '(이름 없음)',
        'de': '(unbenannt)',
        'fr': '(sans nom)',
        'es': '(sin nombre)',
    },
    'No frame falls within the record: check the columns and the offset.': {
        'zh_CN': '没有帧落在记录范围内：请检查所选列和偏移。',
        'zh_TW': '沒有影格落在記錄範圍內：請檢查所選欄與偏移。',
        'ja': '記録の範囲に入るフレームがありません：列とオフセットを確認してください。',
        'ko': '기록 범위에 드는 프레임이 없습니다: 열과 오프셋을 확인하십시오.',
        'de': 'Kein Bild liegt im Aufzeichnungsbereich: Prüfen Sie die Spalten und den Versatz.',
        'fr': "Aucune image ne tombe dans l'enregistrement : vérifiez les colonnes et le décalage.",
        'es': 'Ningún fotograma cae dentro del registro: compruebe las columnas y el desfase.',
    },
    'Frames with a load: %1 of %2.': {
        'zh_CN': '有载荷值的帧：%1 / %2。',
        'zh_TW': '有載荷值的影格：%1 / %2。',
        'ja': '荷重のあるフレーム：%1 / %2。',
        'ko': '하중이 있는 프레임: %1 / %2.',
        'de': 'Bilder mit Kraftwert: %1 von %2.',
        'fr': 'Images avec une charge : %1 sur %2.',
        'es': 'Fotogramas con carga: %1 de %2.',
    },
    'Open Load Data': {
        'zh_CN': '打开载荷数据',
        'zh_TW': '開啟載荷資料',
        'ja': '荷重データを開く',
        'ko': '하중 데이터 열기',
        'de': 'Kraftdaten öffnen',
        'fr': 'Ouvrir les données de charge',
        'es': 'Abrir datos de carga',
    },
    'Could not read %1: %2': {
        'zh_CN': '无法读取 %1：%2',
        'zh_TW': '無法讀取 %1：%2',
        'ja': '%1 を読み込めませんでした：%2',
        'ko': '%1을(를) 읽을 수 없습니다: %2',
        'de': '%1 konnte nicht gelesen werden: %2',
        'fr': 'Impossible de lire %1 : %2',
        'es': 'No se pudo leer %1: %2',
    },
    # ===== Analysis stress-strain view (P6c) ================================
    'Stress–strain': {
        'zh_CN': '应力–应变',
        'zh_TW': '應力–應變',
        'ja': '応力–ひずみ',
        'ko': '응력–변형률',
        'de': 'Spannung–Dehnung',
        'fr': 'Contrainte–déformation',
        'es': 'Tensión–deformación',
    },
    'Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.': {
        'zh_CN': '以所绘物理量为横轴的应力（无 A0 时为载荷）曲线，每个探针一条：配合引伸计为应力–应变曲线，配合裂纹量规为载荷–张开量曲线。需要载荷数据。',
        'zh_TW': '以所繪物理量為橫軸的應力（無 A0 時為載荷）曲線，每個探針一條：搭配引伸計為應力–應變曲線，搭配裂紋量規為載荷–張開量曲線。需要載荷資料。',
        'ja': '表示中の量に対する応力（A0 がない場合は荷重）の曲線を、プローブごとに描きます：伸び計なら応力–ひずみ曲線、き裂ゲージなら荷重–開口量曲線です。荷重データが必要です。',
        'ko': '표시 중인 양에 대한 응력(A0가 없으면 하중) 곡선을 프로브마다 그립니다: 신율계면 응력–변형률, 균열 게이지면 하중–개구 곡선입니다. 하중 데이터가 필요합니다.',
        'de': 'Spannung (ohne A0 die Kraft) über der dargestellten Größe, eine Kurve je Sonde: Spannungs-Dehnungs-Kurve mit einem Dehnungsaufnehmer, Kraft über Rissöffnung mit einer Riss-Messstrecke. Benötigt Kraftdaten.',
        'fr': 'Contrainte (ou charge, sans A0) en fonction de la grandeur tracée, une courbe par sonde : contrainte-déformation avec un extensomètre, charge-ouverture avec une jauge de fissure. Nécessite des données de charge.',
        'es': 'Tensión (o carga, sin A0) frente a la magnitud representada, una curva por sonda: tensión-deformación con un extensómetro, carga-apertura con un calibre de grieta. Requiere datos de carga.',
    },
    "Import the testing machine's load record with Load data… to draw stress-strain curves.": {
        'zh_CN': '请通过“载荷数据…”导入试验机的载荷记录，以绘制应力–应变曲线。',
        'zh_TW': '請透過「載荷資料…」匯入試驗機的載荷記錄，以繪製應力–應變曲線。',
        'ja': '応力–ひずみ曲線を描くには、「荷重データ…」で試験機の荷重記録を読み込んでください。',
        'ko': "응력–변형률 곡선을 그리려면 '하중 데이터…'로 시험기의 하중 기록을 가져오십시오.",
        'de': 'Importieren Sie die Kraftaufzeichnung der Prüfmaschine über „Kraftdaten…“, um Spannungs-Dehnungs-Kurven zu zeichnen.',
        'fr': "Importez l'enregistrement de charge de la machine d'essai via « Données de charge… » pour tracer des courbes contrainte-déformation.",
        'es': 'Importe el registro de carga de la máquina de ensayo con «Datos de carga…» para trazar curvas tensión-deformación.',
    },
}


# -- Numerus-form translations (use <numerusform>…</numerusform>) ------------
# Chinese/Japanese/Korean only need a single form; Romance/Germanic need two;
# in principle some Slavic languages need three, but none of our target langs
# are Slavic, so the two-form layout covers everything.
NUMERUS_TRANSLATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "Import Region of Interest for %n frame(s)": {
        "zh_CN": ("为 %n 帧导入感兴趣区域",),
        "zh_TW": ("為 %n 影格匯入感興趣區域",),
        "ja":    ("%n フレームに関心領域をインポート",),
        "ko":    ("%n 프레임에 관심 영역 가져오기",),
        "de":    ("Region of Interest für %n Bild importieren",
                  "Region of Interest für %n Bilder importieren"),
        "fr":    ("Importer la région d'intérêt pour %n image",
                  "Importer la région d'intérêt pour %n images"),
        "es":    ("Importar región de interés para %n fotograma",
                  "Importar región de interés para %n fotogramas"),
    },
    "Delete %n image(s)": {
        "zh_CN": ("删除 %n 张图像",),
        "zh_TW": ("刪除 %n 張影像",),
        "ja":    ("%n 個の画像を削除",),
        "ko":    ("%n 개 이미지 삭제",),
        "de":    ("%n Bild löschen", "%n Bilder löschen"),
        "fr":    ("Supprimer %n image", "Supprimer %n images"),
        "es":    ("Eliminar %n imagen", "Eliminar %n imágenes"),
    },
    "Select %n Mask File(s)": {
        "zh_CN": ("选择 %n 个蒙版文件",),
        "zh_TW": ("選擇 %n 個遮罩檔案",),
        "ja":    ("%n 個のマスクファイルを選択",),
        "ko":    ("%n 개 마스크 파일 선택",),
        "de":    ("%n Maskendatei auswählen", "%n Maskendateien auswählen"),
        "fr":    ("Sélectionner %n fichier de masque",
                  "Sélectionner %n fichiers de masque"),
        "es":    ("Seleccionar %n archivo de máscara",
                  "Seleccionar %n archivos de máscara"),
    },
    "Imported Region of Interest for %n frame(s)": {
        "zh_CN": ("为 %n 帧导入了感兴趣区域",),
        "zh_TW": ("為 %n 影格匯入了感興趣區域",),
        "ja":    ("%n フレームに関心領域をインポートしました",),
        "ko":    ("%n 프레임에 관심 영역 가져옴",),
        "de":    ("Region of Interest für %n Bild importiert",
                  "Region of Interest für %n Bilder importiert"),
        "fr":    ("Région d'intérêt importée pour %n image",
                  "Région d'intérêt importée pour %n images"),
        "es":    ("Región de interés importada para %n fotograma",
                  "Región de interés importada para %n fotogramas"),
    },
    "  %n frame(s) with custom ROI masks": {
        "zh_CN": ("  %n 帧使用自定义感兴趣区域蒙版",),
        "zh_TW": ("  %n 影格使用自訂感興趣區域遮罩",),
        "ja":    ("  %n 個のフレームでカスタム ROI マスクを使用",),
        "ko":    ("  %n 개 프레임에서 사용자 지정 ROI 마스크 사용",),
        "de":    ("  %n Bild mit benutzerdefinierter ROI-Maske",
                  "  %n Bilder mit benutzerdefinierten ROI-Masken"),
        "fr":    ("  %n image avec un masque ROI personnalisé",
                  "  %n images avec des masques ROI personnalisés"),
        "es":    ("  %n fotograma con máscara ROI personalizada",
                  "  %n fotogramas con máscaras ROI personalizadas"),
    },
    "Results received: %n frame(s)": {
        "zh_CN": ("已收到结果：%n 帧",),
        "zh_TW": ("已收到結果：%n 影格",),
        "ja":    ("結果を受信：%n フレーム",),
        "ko":    ("결과 수신: %n 프레임",),
        "de":    ("Ergebnisse empfangen: %n Bild",
                  "Ergebnisse empfangen: %n Bilder"),
        "fr":    ("Résultats reçus : %n image",
                  "Résultats reçus : %n images"),
        "es":    ("Resultados recibidos: %n fotograma",
                  "Resultados recibidos: %n fotogramas"),
    },
    "Exported %n animation(s) → %1": {
        "zh_CN": ("已导出 %n 个动画 → %1",),
        "zh_TW": ("已匯出 %n 個動畫 → %1",),
        "ja":    ("%n 個のアニメーションをエクスポートしました → %1",),
        "ko":    ("%n 개 애니메이션 내보냄 → %1",),
        "de":    ("%n Animation exportiert → %1",
                  "%n Animationen exportiert → %1"),
        "fr":    ("%n animation exportée → %1",
                  "%n animations exportées → %1"),
        "es":    ("Exportada %n animación → %1",
                  "Exportadas %n animaciones → %1"),
    },

    # zh_CN-only numerus entries for batch-import dialog (size pre-scan
    # warning + completion log). Other languages await contributors.
    "%n mask(s) have mismatched sizes and are disabled.": {
        "zh_CN": ("%n 个掩模尺寸不匹配，已禁用。",),
    },
    "Batch import: %n mask(s) loaded": {
        "zh_CN": ("批量导入：已加载 %n 个掩模",),
    },

    # === Post-processing analysis tab ====================================

    # --- placement tools --------------------------------------------------

    # --- probe list -------------------------------------------------------

    # --- chart controls ---------------------------------------------------

    # --- chart messages ---------------------------------------------------

    # --- export -----------------------------------------------------------

    # --- crash reporting (Application) -------------------------------------

    # --- kernel warm-up (Application) --------------------------------------

    # --- export failure feedback (ExportDialog) ----------------------------

    # --- file dialogs that were previously raw literals --------------------
}


# ---------- .ts editing ----------------------------------------------------

MESSAGE_BLOCK = re.compile(
    r"(<message[^>]*>)(.*?)(</message>)", re.DOTALL
)

SOURCE_INNER = re.compile(r"<source>(.*?)</source>", re.DOTALL)

UNFIN_TRANSLATION = re.compile(
    r"<translation(?:\s+type=\"unfinished\")?\s*>(.*?)</translation>",
    re.DOTALL,
)
UNFIN_NUMERUS = re.compile(
    r"<translation\s+type=\"unfinished\">\s*((?:<numerusform></numerusform>\s*)+)</translation>",
    re.DOTALL,
)


def check_tables() -> None:
    """Refuse a malformed table instead of writing garbage into the catalogs.

    ``NUMERUS_TRANSLATIONS`` values must be sequences of plural forms. A plain
    string filed there is indexed like a sequence, and ``forms[-1]`` is its
    last character: seventy entries shipped that way ("n" as the German title
    of the fatal-error dialog) while every coverage check passed, because a
    one-character translation still counts as translated.
    """
    for source, per_lang in NUMERUS_TRANSLATIONS.items():
        for lang, forms in per_lang.items():
            if isinstance(forms, str) or not all(isinstance(f, str) for f in forms):
                raise TypeError(
                    f"NUMERUS_TRANSLATIONS[{source!r}][{lang!r}] must be a "
                    f"tuple of plural forms, got {forms!r}. A single string "
                    "belongs in TRANSLATIONS; filed here, only its last "
                    "character would be written."
                )
    for source, per_lang in TRANSLATIONS.items():
        for lang, text in per_lang.items():
            if not isinstance(text, str):
                raise TypeError(
                    f"TRANSLATIONS[{source!r}][{lang!r}] must be a string, "
                    f"got {text!r}. Plural forms belong in NUMERUS_TRANSLATIONS."
                )


def fill_ts(lang: str) -> tuple[int, int]:
    """Return (filled, skipped) count for a single language."""
    check_tables()
    ts_path = TS_DIR / f"al_dic_{lang}.ts"
    text = ts_path.read_text(encoding="utf-8")
    filled = 0
    skipped = 0

    def replace_block(match: re.Match[str]) -> str:
        nonlocal filled, skipped
        open_tag, body, close_tag = match.group(1), match.group(2), match.group(3)
        src_m = SOURCE_INNER.search(body)
        if src_m is None:
            return match.group(0)
        source = unescape(src_m.group(1))

        # --- numerusform (plural) entries ---
        if 'numerus="yes"' in open_tag and source in NUMERUS_TRANSLATIONS:
            forms = NUMERUS_TRANSLATIONS[source].get(lang)
            if forms is None:
                skipped += 1
                return match.group(0)
            joined = "\n".join(
                f"            <numerusform>{escape(f)}</numerusform>"
                for f in forms
            )
            new_translation = (
                f"<translation>\n{joined}\n        </translation>"
            )
            new_body = re.sub(
                r"<translation[^>]*>\s*(?:<numerusform></numerusform>\s*)+</translation>",
                new_translation,
                body,
                count=1,
                flags=re.DOTALL,
            )
            if new_body == body:
                skipped += 1
                return match.group(0)
            filled += 1
            return open_tag + new_body + close_tag

        # --- simple unfinished entries ---
        if 'type="unfinished"' not in body:
            return match.group(0)

        entry = TRANSLATIONS.get(source)
        # Fall back to the first numerusform if lupdate didn't tag the
        # message as numerus="yes". Qt still substitutes %n → count,
        # we just miss the plural-form branching — which is acceptable
        # because %n is only interesting for >1 case anyway.
        if entry is None and source in NUMERUS_TRANSLATIONS:
            forms = NUMERUS_TRANSLATIONS[source].get(lang)
            if forms:
                entry = {lang: forms[-1]}  # plural form (or singular for CJK)
        if entry is None:
            skipped += 1
            return match.group(0)
        rendering = entry.get(lang)
        if rendering is None:
            skipped += 1
            return match.group(0)

        new_translation = f"<translation>{escape(rendering)}</translation>"
        new_body = re.sub(
            r'<translation\s+type="unfinished">.*?</translation>',
            new_translation,
            body,
            count=1,
            flags=re.DOTALL,
        )
        if new_body == body:
            skipped += 1
            return match.group(0)
        filled += 1
        return open_tag + new_body + close_tag

    new_text = MESSAGE_BLOCK.sub(replace_block, text)
    if filled:
        ts_path.write_text(new_text, encoding="utf-8")
    return filled, skipped


def main() -> None:
    for lang in LANGUAGES:
        filled, skipped = fill_ts(lang)
        print(f"  {lang:<6} filled={filled:<4} skipped={skipped}")


if __name__ == "__main__":
    main()
