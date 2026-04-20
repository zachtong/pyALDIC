# pyALDIC Translation Glossary

Fixed translations for domain-specific DIC terminology. This file is
the single source of truth for all languages. When translating `.ts`
files, consult this table first; when a term is missing and likely to
recur, add it here in a PR rather than improvising ad-hoc translations.

New languages: add a column.
New terms: add a row, fill at least English + 简体中文.

---

## Do NOT translate (keep English in every locale)

Proper nouns, method names, file-format tokens, and established
abbreviations stay English across all languages:

| Category | Tokens |
|---|---|
| Brand / method | `pyALDIC`, `AL-DIC`, `IC-GN`, `ADMM`, `FEM`, `Q8` |
| Algorithm | `FFT`, `NCC`, `BFS`, `Quadtree` |
| File formats | `MAT`, `NPZ`, `CSV`, `PNG`, `GIF`, `MP4`, `PDF`, `BMP`, `TIFF` |
| Tech abbreviations | `GUI`, `CLI`, `CPU`, `GPU`, `CI`, `ROI`, `POI`, `RMSE` |
| Library names | `PySide6`, `NumPy`, `SciPy`, `Numba`, `Matplotlib` |

---

## Core DIC terminology

| English | 简体中文 | 繁體中文 | 日本語 |
|---|---|---|---|
| Digital Image Correlation | 数字图像相关 | 數字圖像相關 | デジタル画像相関 |
| Subset | 子集 | 子集 | サブセット |
| Subset size | 子集尺寸 | 子集尺寸 | サブセットサイズ |
| Subset step | 子集步长 | 子集步長 | サブセットステップ |
| Search range | 搜索范围 | 搜尋範圍 | 探索範囲 |
| Shape function | 形函数 | 形函數 | 形状関数 |
| Reference frame | 参考帧 | 參考幀 | 参照フレーム |
| Deformed frame | 变形帧 | 變形幀 | 変形フレーム |
| Displacement field | 位移场 | 位移場 | 変位場 |
| Strain field | 应变场 | 應變場 | 歪み場 |
| Correlation | 相关 | 相關 | 相関 |
| Mesh | 网格 | 網格 | メッシュ |
| Adaptive mesh | 自适应网格 | 自適應網格 | 適応メッシュ |
| Quadtree refinement | 四叉树加密 | 四叉樹加密 | 四分木細分化 |
| Mask | 掩模 | 遮罩 | マスク |
| Region of Interest (ROI) | 感兴趣区域 | 感興趣區域 | 関心領域 |
| Seed / Starting point | 种子点 | 種子點 | シード点 |
| Seed propagation | 种子传播 | 種子傳播 | シード伝播 |
| Incremental tracking | 增量追踪 | 增量追蹤 | 逐次追跡 |
| Accumulative tracking | 累积追踪 | 累積追蹤 | 累積追跡 |
| Augmented Lagrangian | 增广拉格朗日 | 增廣拉格朗日 | 拡張ラグランジュ |
| Initial guess | 初始猜测 | 初始猜測 | 初期推定 |
| Convergence | 收敛 | 收斂 | 収束 |
| Iteration | 迭代 | 迭代 | 反復 |

---

## GUI common verbs / labels

| English | 简体中文 | 繁體中文 | 日本語 |
|---|---|---|---|
| Run | 运行 | 執行 | 実行 |
| Cancel | 取消 | 取消 | キャンセル |
| OK | 确定 | 確定 | OK |
| Apply | 应用 | 套用 | 適用 |
| Save | 保存 | 儲存 | 保存 |
| Load | 加载 | 載入 | 読み込み |
| Import | 导入 | 匯入 | インポート |
| Export | 导出 | 匯出 | エクスポート |
| Refine | 加密 | 加密 | 細分化 |
| Reset | 重置 | 重設 | リセット |
| Clear | 清除 | 清除 | クリア |
| Open | 打开 | 開啟 | 開く |
| Close | 关闭 | 關閉 | 閉じる |
| Settings | 设置 | 設定 | 設定 |
| Preferences | 首选项 | 偏好設定 | 環境設定 |
| File | 文件 | 檔案 | ファイル |
| Edit | 编辑 | 編輯 | 編集 |
| View | 视图 | 檢視 | 表示 |
| Help | 帮助 | 說明 | ヘルプ |
| About | 关于 | 關於 | について |
| Browse... | 浏览… | 瀏覽… | 参照… |

---

## Status / progress terms

| English | 简体中文 | 繁體中文 | 日本語 |
|---|---|---|---|
| Ready | 就绪 | 就緒 | 準備完了 |
| Running | 运行中 | 執行中 | 実行中 |
| Paused | 已暂停 | 已暫停 | 一時停止中 |
| Completed | 已完成 | 已完成 | 完了 |
| Failed | 失败 | 失敗 | 失敗 |
| Frame | 帧 | 幀 | フレーム |
| Elapsed | 已用时间 | 已用時間 | 経過時間 |
| Remaining | 剩余时间 | 剩餘時間 | 残り時間 |
| Progress | 进度 | 進度 | 進捗 |

---

## Review / QA checklist when adding a language

1. Every row in the tables above has the column filled.
2. Button labels stay **short** — aim for ≤6 CJK characters or ≤12
   Latin characters; longer labels get truncated in narrow widgets.
3. Technical tokens from the "Do NOT translate" section are kept as
   literal English inside translated strings.
4. Qt `%1 %2 %n` placeholders are preserved verbatim.
5. Ampersand mnemonics (`&Save` → Alt+S) in English source are
   dropped in translations; quick-keys come from `QAction`-level
   shortcuts instead.
