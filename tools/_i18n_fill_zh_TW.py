"""Fill al_dic_zh_TW.ts from zh_CN via OpenCC s2tw + Taiwan vocabulary.

Strategy
--------
1. Start from the already-complete zh_CN translation dict.
2. Convert each value with OpenCC's ``s2tw`` (character-level Simplified
   to Traditional) — this is safer than ``s2twp`` which sometimes maps
   "图像" to the incorrect "影象" (uses 象 for "elephant" instead of
   像 for "image").
3. Apply a manual Taiwan-vocabulary override dict on top: CN mainland
   terms that diverge between PRC and Taiwan (软件 -> 軟體, 文件 ->
   檔案, 信息 -> 資訊, etc.).

DIC technical vocabulary (位移 / 應變 / 子集 / 網格 / 像素 / 梯度 /
收斂) is shared between CN and TW and survives conversion untouched.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from opencc import OpenCC

TS = Path(__file__).resolve().parents[1] / \
    "src" / "al_dic" / "i18n" / "source" / "al_dic_zh_TW.ts"

# Lazy-load the zh_CN fill module (sibling file) without polluting imports.
_SPEC = importlib.util.spec_from_file_location(
    "_i18n_fill_zh_CN",
    Path(__file__).parent / "_i18n_fill_zh_CN.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_ZH_CN: dict[str, str] = _MOD.TRANSLATIONS

_CC = OpenCC("s2tw")  # char-only conversion, phrase overrides below

# CN -> TW phrase substitutions applied AFTER char conversion. Kept
# narrow to avoid over-correction: only terms where PRC and Taiwan
# unambiguously diverge in current technical / software usage.
_TW_OVERRIDES: dict[str, str] = {
    # Software / UI
    "軟件": "軟體",
    "設置": "設定",
    "默認": "預設",
    "程序": "程式",
    "信息": "資訊",
    "文件夾": "資料夾",
    "文件": "檔案",
    "打開": "開啟",
    "加載": "載入",
    "下載": "下載",  # same
    "上傳": "上傳",  # same
    "導入": "匯入",
    "導出": "匯出",
    "保存": "儲存",
    "刪除": "刪除",  # same
    "菜單": "選單",
    "鼠標": "滑鼠",
    "視頻": "影片",
    "網絡": "網路",
    "登錄": "登入",
    "註銷": "登出",
    "打印": "列印",
    "搜索": "搜尋",
    "反饋": "回饋",
    # Data / formats — keep file extensions and acronyms English.
    "數據": "資料",
    "列表": "清單",
    # Images / media
    "圖像": "影像",  # Taiwan prefers 影像 for digital image
    # UI-specific buttons / states
    "確認": "確認",  # same
    "退出": "離開",
    # Engineering terms — usually identical, kept explicit for clarity
    "子集": "子集",
    "位移": "位移",
    "應變": "應變",
    "網格": "網格",
    "像素": "像素",
    "幀": "幀",
    "參考": "參考",
    "梯度": "梯度",
    "收斂": "收斂",
    "迭代": "迭代",
}


def _to_tw(text: str) -> str:
    t = _CC.convert(text)
    for cn, tw in _TW_OVERRIDES.items():
        t = t.replace(cn, tw)
    return t


def _apply(ts_path: Path, mapping: dict[str, str]) -> tuple[int, list[str]]:
    """Fill every ``<translation type="unfinished">`` whose source
    matches a key in ``mapping``. Returns (n_filled, miss_list)."""
    content = ts_path.read_text(encoding="utf-8")
    filled = 0
    miss: list[str] = []

    def _replace(match: re.Match) -> str:
        nonlocal filled
        source = match.group(1)
        # Undo lupdate's XML escaping on the key.
        key = (source
               .replace("&amp;", "&")
               .replace("&lt;", "<")
               .replace("&gt;", ">")
               .replace("&quot;", '"')
               .replace("&apos;", "'"))
        target = mapping.get(key)
        if target is None:
            miss.append(key)
            return match.group(0)
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
        _replace,
        content,
        flags=re.DOTALL,
    )
    ts_path.write_text(new_content, encoding="utf-8")
    return filled, miss


def main() -> None:
    # Build zh_TW translation dict from zh_CN + OpenCC + overrides.
    tw_map: dict[str, str] = {k: _to_tw(v) for k, v in _ZH_CN.items()}

    filled, miss = _apply(TS, tw_map)
    print(f"[zh_TW] filled: {filled}  |  missing: {len(miss)}")
    for s in miss[:15]:
        print(f"  MISS: {s!r}")


if __name__ == "__main__":
    main()
