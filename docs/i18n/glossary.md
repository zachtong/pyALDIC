# pyALDIC Translation Glossary

> **Status**: locked for v0.x. Living document — propose changes via
> issue / PR rather than editing in-place. Once a term ships in a
> released `.qm` catalog, changing it requires bumping the catalog
> version (otherwise users see mixed translations).

Fixed translations for domain-specific DIC terminology and high-frequency
GUI verbs. Single source of truth for all 7 non-English languages.

When translating `.ts` entries (or filling
`tools/fill_translations.py`):

1. Search this file first.
2. If the term is here, copy verbatim. Do not improvise variants.
3. If missing **and likely to recur**, add a row before translating.
4. If missing and one-shot, translate inline but flag with a `# TODO:
   glossary?` comment for later canonicalization.

New languages: add a column to every table.
New terms: add a row, fill at least English + zh_CN.

---

## Do NOT translate (keep English in every locale)

Proper nouns, method names, file-format tokens, math symbols, and
established abbreviations stay literal English across all 8 languages:

| Category | Tokens |
|---|---|
| Brand / method | `pyALDIC`, `AL-DIC`, `IC-GN`, `ADMM`, `FEM`, `Q8`, `DIC` |
| Algorithm | `FFT`, `NCC`, `BFS`, `Quadtree`, `IDW`, `VSG` |
| File formats | `MAT`, `NPZ`, `CSV`, `PNG`, `GIF`, `MP4`, `PDF`, `BMP`, `TIFF`, `JPEG`, `HTML` |
| Tech abbreviations | `GUI`, `CLI`, `CPU`, `GPU`, `CI`, `ROI`, `POI`, `RMSE`, `DPI`, `FPS` |
| Library names | `PySide6`, `NumPy`, `SciPy`, `Numba`, `Matplotlib` |
| Math symbols | `α β σ ε γ ω π τ θ × ÷ ± ≤ ≥ ⚠ ∞` |
| Strain components | `εxx εyy εxy ε₁ ε₂ γ max von Mises ω rot` |
| Colormap names | `jet viridis turbo coolwarm plasma RdBu_r seismic inferno black_rainbow` |

**Display-facing exceptions**:
- `ADMM Iterations` → re-labeled `AL-DIC Iterations` for users (see
  CLAUDE.md whitelist note). The internal state variable
  `admm_max_iter` is unchanged.

---

## Core DIC terminology

| English | zh_CN | zh_TW | ja | ko | de | fr | es |
|---|---|---|---|---|---|---|---|
| Digital Image Correlation | 数字图像相关 | 數字圖像相關 | デジタル画像相関 | 디지털 이미지 상관법 | Digitale Bildkorrelation | Corrélation d'images numériques | Correlación digital de imágenes |
| Subset | 子集 | 子集 | サブセット | 서브셋 | Subset | Subset | Subset |
| Subset size | 子集尺寸 | 子集尺寸 | サブセットサイズ | 서브셋 크기 | Subset-Größe | Taille du subset | Tamaño del subset |
| Subset step | 子集步长 | 子集步長 | サブセットステップ | 서브셋 스텝 | Subset-Schritt | Pas du subset | Paso del subset |
| Search range | 搜索范围 | 搜尋範圍 | 探索範囲 | 검색 범위 | Suchbereich | Plage de recherche | Rango de búsqueda |
| Shape function | 形函数 | 形函數 | 形状関数 | 형상 함수 | Formfunktion | Fonction de forme | Función de forma |
| Reference frame | 参考帧 | 參考幀 | 参照フレーム | 참조 프레임 | Referenzbild | Image de référence | Fotograma de referencia |
| Deformed frame | 变形帧 | 變形幀 | 変形フレーム | 변형된 프레임 | Verformtes Bild | Image déformée | Fotograma deformado |
| Displacement field | 位移场 | 位移場 | 変位場 | 변위장 | Verschiebungsfeld | Champ de déplacement | Campo de desplazamiento |
| Strain field | 应变场 | 應變場 | ひずみ場 | 변형률장 | Dehnungsfeld | Champ de déformation | Campo de deformación |
| Strain | 应变 | 應變 | ひずみ | 변형률 | Dehnung | Déformation | Deformación |
| Displacement | 位移 | 位移 | 変位 | 변위 | Verschiebung | Déplacement | Desplazamiento |
| Velocity | 速度 | 速度 | 速度 | 속도 | Geschwindigkeit | Vitesse | Velocidad |
| Correlation | 相关 | 相關 | 相関 | 상관 | Korrelation | Corrélation | Correlación |
| Mesh | 网格 | 網格 | メッシュ | 메시 | Netz | Maillage | Malla |
| Adaptive mesh | 自适应网格 | 自適應網格 | 適応メッシュ | 적응형 메시 | Adaptives Netz | Maillage adaptatif | Malla adaptativa |
| Quadtree refinement | 四叉树加密 | 四叉樹加密 | 四分木細分化 | 쿼드트리 세분화 | Quadtree-Verfeinerung | Raffinement quadtree | Refinamiento quadtree |
| Mesh refinement | 网格加密 | 網格加密 | メッシュ細分化 | 메시 세분화 | Netzverfeinerung | Raffinement de maillage | Refinamiento de malla |
| Mask | 掩模 | 遮罩 | マスク | 마스크 | Maske | Masque | Máscara |
| Region of Interest (ROI) | 感兴趣区域 | 感興趣區域 | 関心領域 | 관심 영역 | Region of Interest | Région d'intérêt | Región de interés |
| Seed / Starting point | 种子点 | 種子點 | シード点 | 시드 점 | Startpunkt | Point de départ | Punto inicial |
| Seed propagation | 种子传播 | 種子傳播 | シード伝播 | 시드 전파 | Seed-Propagation | Propagation des seeds | Propagación de seeds |
| Incremental tracking | 增量追踪 | 增量追蹤 | 逐次追跡 | 점진적 추적 | Inkrementelles Tracking | Suivi incrémental | Seguimiento incremental |
| Accumulative tracking | 累积追踪 | 累積追蹤 | 累積追跡 | 누적 추적 | Akkumulatives Tracking | Suivi cumulatif | Seguimiento acumulativo |
| Augmented Lagrangian | 增广拉格朗日 | 增廣拉格朗日 | 拡張ラグランジュ | 증강 라그랑주 | Erweiterte Lagrange | Lagrangien augmenté | Lagrangiano aumentado |
| Initial guess | 初始猜测 | 初始猜測 | 初期推定 | 초기 추정 | Anfangsschätzung | Estimation initiale | Estimación inicial |
| Convergence | 收敛 | 收斂 | 収束 | 수렴 | Konvergenz | Convergence | Convergencia |
| Iteration | 迭代 | 迭代 | 反復 | 반복 | Iteration | Itération | Iteración |
| Plane fitting | 平面拟合 | 平面擬合 | 平面フィット | 평면 피팅 | Ebenenanpassung | Ajustement de plan | Ajuste de plano |
| FEM nodal | FEM 节点 | FEM 節點 | FEM ノード | FEM 노드 | FEM-Knoten | FEM nodal | FEM nodal |
| Smoothing | 平滑 | 平滑 | 平滑化 | 평활화 | Glättung | Lissage | Suavizado |
| Outlier | 异常值 | 異常值 | 外れ値 | 이상치 | Ausreißer | Valeur aberrante | Valor atípico |
| Refinement level | 加密级别 | 加密級別 | 細分化レベル | 세분화 레벨 | Verfeinerungsstufe | Niveau de raffinement | Nivel de refinamiento |
| Pixel size | 像素尺寸 | 像素尺寸 | ピクセルサイズ | 픽셀 크기 | Pixelgröße | Taille de pixel | Tamaño de píxel |
| Frame rate | 帧率 | 影格率 | フレームレート | 프레임 속도 | Bildrate | Fréquence d'images | Velocidad de fotogramas |
| Colormap | 颜色映射 | 色彩對映 | カラーマップ | 색상 맵 | Farbskala | Palette | Mapa de colores |
| Colorbar | 色条 | 色條 | カラーバー | 컬러바 | Farbleiste | Barre de couleur | Barra de color |
| Opacity | 不透明度 | 不透明度 | 不透明度 | 불투명도 | Deckkraft | Opacité | Opacidad |
| Range | 范围 | 範圍 | 範囲 | 범위 | Bereich | Plage | Rango |

---

## Post-processing analysis (probes)

| English | zh_CN | zh_TW | ja | ko | de | fr | es |
|---|---|---|---|---|---|---|---|
| Probe | 探针 | 探針 | プローブ | 프로브 | Sonde | Sonde | Sonda |
| Region (probe type) | 区域 | 區域 | 領域 | 영역 | Bereich | Région | Región |
| Virtual extensometer | 虚拟引伸计 | 虛擬引伸計 | 仮想伸び計 | 가상 신율계 | Virtueller Dehnungsaufnehmer | Extensomètre virtuel | Extensómetro virtual |
| Gauge (line reading) | 量规 | 量規 | ゲージ | 게이지 | Messstrecke | Jauge | Calibre |
| Crack | 裂纹 | 裂紋 | き裂 | 균열 | Riss | Fissure | Grieta |
| Crack opening | 裂纹张开位移 | 裂紋張開位移 | き裂開口変位 | 균열 개구 변위 | Rissöffnung | Ouverture de fissure | Apertura de grieta |
| Crack gauge | 裂纹量规 | 裂紋量規 | き裂ゲージ | 균열 게이지 | Riss-Messstrecke | Jauge de fissure | Calibre de grieta |

---

## GUI common verbs / labels

| English | zh_CN | zh_TW | ja | ko | de | fr | es |
|---|---|---|---|---|---|---|---|
| Run | 运行 | 執行 | 実行 | 실행 | Ausführen | Exécuter | Ejecutar |
| Cancel | 取消 | 取消 | キャンセル | 취소 | Abbrechen | Annuler | Cancelar |
| OK | 确定 | 確定 | OK | 확인 | OK | OK | Aceptar |
| Apply | 应用 | 套用 | 適用 | 적용 | Anwenden | Appliquer | Aplicar |
| Save | 保存 | 儲存 | 保存 | 저장 | Speichern | Enregistrer | Guardar |
| Load | 加载 | 載入 | 読み込み | 불러오기 | Laden | Charger | Cargar |
| Import | 导入 | 匯入 | インポート | 가져오기 | Importieren | Importer | Importar |
| Export | 导出 | 匯出 | エクスポート | 내보내기 | Exportieren | Exporter | Exportar |
| Refine | 加密 | 加密 | 細分化 | 세분화 | Verfeinern | Raffiner | Refinar |
| Reset | 重置 | 重設 | リセット | 재설정 | Zurücksetzen | Réinitialiser | Restablecer |
| Clear | 清除 | 清除 | クリア | 지우기 | Löschen | Effacer | Borrar |
| Open | 打开 | 開啟 | 開く | 열기 | Öffnen | Ouvrir | Abrir |
| Close | 关闭 | 關閉 | 閉じる | 닫기 | Schließen | Fermer | Cerrar |
| Settings | 设置 | 設定 | 設定 | 설정 | Einstellungen | Paramètres | Ajustes |
| Preferences | 首选项 | 偏好設定 | 環境設定 | 환경 설정 | Voreinstellungen | Préférences | Preferencias |
| File | 文件 | 檔案 | ファイル | 파일 | Datei | Fichier | Archivo |
| Filename | 文件名 | 檔名 | ファイル名 | 파일 이름 | Dateiname | Nom de fichier | Nombre de archivo |
| Edit | 编辑 | 編輯 | 編集 | 편집 | Bearbeiten | Édition | Editar |
| View | 视图 | 檢視 | 表示 | 보기 | Ansicht | Affichage | Ver |
| Help | 帮助 | 說明 | ヘルプ | 도움말 | Hilfe | Aide | Ayuda |
| About | 关于 | 關於 | について | 정보 | Über | À propos | Acerca de |
| Browse... | 浏览… | 瀏覽… | 参照… | 찾아보기… | Durchsuchen… | Parcourir… | Examinar… |
| Generate | 生成 | 產生 | 生成 | 생성 | Erstellen | Générer | Generar |
| Auto | 自动 | 自動 | 自動 | 자동 | Auto | Auto | Auto |
| Min | 最小 | 最小 | 最小 | 최소 | Min | Min | Mín |
| Max | 最大 | 最大 | 最大 | 최대 | Max | Max | Máx |
| Format | 格式 | 格式 | 形式 | 형식 | Format | Format | Formato |
| Field | 字段 | 欄位 | フィールド | 필드 | Feld | Champ | Campo |
| All | 全部 | 全部 | すべて | 모두 | Alle | Tous | Todos |
| None | 无 | 無 | なし | 없음 | Keine | Aucun | Ninguno |

---

## Status / progress terms

| English | zh_CN | zh_TW | ja | ko | de | fr | es |
|---|---|---|---|---|---|---|---|
| Ready | 就绪 | 就緒 | 準備完了 | 준비됨 | Bereit | Prêt | Listo |
| Running | 运行中 | 執行中 | 実行中 | 실행 중 | Läuft | En cours | En ejecución |
| Paused | 已暂停 | 已暫停 | 一時停止中 | 일시 정지됨 | Angehalten | En pause | En pausa |
| Completed | 已完成 | 已完成 | 完了 | 완료됨 | Abgeschlossen | Terminé | Completado |
| Failed | 失败 | 失敗 | 失敗 | 실패 | Fehlgeschlagen | Échec | Fallido |
| Starting… | 开始中… | 開始中… | 開始中… | 시작 중… | Wird gestartet… | Démarrage… | Iniciando… |
| Frame | 帧 | 影格 | フレーム | 프레임 | Bild | Image | Fotograma |
| Elapsed | 已用时间 | 已用時間 | 経過時間 | 경과 시간 | Verstrichen | Écoulé | Transcurrido |
| Remaining | 剩余时间 | 剩餘時間 | 残り時間 | 남은 시간 | Verbleibend | Restant | Restante |
| Progress | 进度 | 進度 | 進捗 | 진행률 | Fortschritt | Progression | Progreso |
| Error | 错误 | 錯誤 | エラー | 오류 | Fehler | Erreur | Error |
| Warning | 警告 | 警告 | 警告 | 경고 | Warnung | Avertissement | Advertencia |

---

## Severity labels (mesh refinement, smoothing presets)

| English | zh_CN | zh_TW | ja | ko | de | fr | es |
|---|---|---|---|---|---|---|---|
| Off | 关闭 | 關閉 | オフ | 끔 | Aus | Désactivé | Desactivado |
| Light | 轻度 | 輕度 | 軽度 | 약함 | Leicht | Léger | Ligero |
| Medium | 中等 | 中等 | 中程度 | 중간 | Mittel | Moyen | Medio |
| Heavy / Strong | 强 | 強 | 強 | 강함 | Stark | Fort | Fuerte |
| Extra Heavy | 超强 | 超強 | 最強 | 매우 강함 | Sehr stark | Très fort | Muy fuerte |
| Ultra | 极限 | 極限 | 極限 | 극강 | Ultra | Ultra | Ultra |

---

## Review / QA checklist when adding a language

1. Every row above has the new column filled.
2. Button labels stay **short** — aim for ≤6 CJK characters or ≤12
   Latin characters; longer labels get truncated in narrow widgets.
3. Tokens from the **Do NOT translate** section stay literal English.
4. Qt `%1 %2 %n` placeholders preserved verbatim.
5. Ampersand mnemonics (`&Save` → Alt+S) dropped in translations;
   keyboard shortcuts come from `QAction.setShortcut(QKeySequence.X)`.
6. Numeric formatting respects the locale (German `3,14` vs `3.14`).
   Route through `al_dic.utils.locale_format.format_number`.
7. CJK font fallback: any matplotlib figure must call
   `configure_matplotlib_fonts()` first or it renders tofu.

---

## How to propose a glossary change

1. Open a GitHub issue tagged `i18n-glossary` describing the term and
   proposed translation(s).
2. If the term is **new** (no existing translation), maintainer can
   merge after one round of review.
3. If the term **changes** existing translation, the change must
   coincide with a major-or-minor version bump (`v0.5.0`, not
   `v0.4.x`) so users don't see mixed translations across an upgrade.
4. After merge: run `python tools/i18n.py extract` and update every
   `.ts` file (and `tools/fill_translations.py`) to use the new term.
