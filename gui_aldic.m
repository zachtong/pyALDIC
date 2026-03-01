function gui_aldic()
%gui_aldic  Launch the STAQ-DIC graphical user interface.
%
%   gui_aldic() opens a programmatic uifigure GUI for running the AL-DIC
%   pipeline with adaptive quadtree mesh. Provides image/mask browsing,
%   ROI selection, parameter editing, progress tracking, and results
%   visualization.
%
% ----------------------------------------------
% Original AL-DIC by Jin Yang, PhD @Caltech
% GUI by Zach Tong, 2026-02
% ==============================================

    %% Environment setup
    setenv('MW_MINGW64_LOC', 'C:\\TDM-GCC-64');
    try mex -O -outdir ./third_party ./third_party/ba_interp2_spline.cpp; catch; end
    addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
            './third_party','./third_party/rbfinterp');

    %% App state
    app = struct();
    app.file_name = {};
    app.Img = {};
    app.ImgMask = {};
    app.mask_file_name = {};
    app.results = [];
    app.stopRequested = false;
    app.isRunning = false;

    % Region state (composable operation list)
    app.regionOps = {};   % ordered cell array of operation structs
                          % .type: 'rect'|'polygon'|'circle'|'cut_polygon'|'imported'
                          % .data: shape-specific (display coords or code-space masks)
    app.roiGridx = [];    % [xmin, xmax] code space
    app.roiGridy = [];    % [ymin, ymax] code space
    app.advancedVisible = false;
    app.lastImgDir = pwd;     % session path memory (reset on GUI restart)
    app.lastMaskDir = pwd;

    %% Create main figure (HandleVisibility='off' to survive close all)
    app.fig = uifigure('Name', 'STAQ-DIC GUI', ...
        'Position', [100 100 1100 700], ...
        'HandleVisibility', 'off', ...
        'CloseRequestFcn', @onClose);

    %% Layout: left panel (280px) + right panel (flexible)
    mainGrid = uigridlayout(app.fig, [1, 2]);
    mainGrid.ColumnWidth = {280, '1x'};
    mainGrid.Padding = [5 5 5 5];
    mainGrid.ColumnSpacing = 5;

    % --- Left panel ---
    leftPanel = uigridlayout(mainGrid, [7, 1]);
    leftPanel.RowHeight = {130, 130, '1x', 35, 35, 35, 35};
    leftPanel.Padding = [0 0 0 0];
    leftPanel.RowSpacing = 5;

    % --- Right panel ---
    rightPanel = uigridlayout(mainGrid, [4, 1]);
    rightPanel.RowHeight = {'1x', 35, 55, 160};
    rightPanel.Padding = [0 0 0 0];
    rightPanel.RowSpacing = 5;

    %% ====== LEFT: Images section ======
    imgPanel = uipanel(leftPanel, 'Title', 'Images');
    imgGrid = uigridlayout(imgPanel, [2, 1]);
    imgGrid.RowHeight = {25, '1x'};
    imgGrid.Padding = [5 5 5 5];

    app.btnLoadImages = uibutton(imgGrid, 'Text', 'Browse Folder...', ...
        'ButtonPushedFcn', @onLoadImages);
    app.lstImages = uilistbox(imgGrid, 'Items', {}, ...
        'ValueChangedFcn', @onImageSelected);

    %% ====== LEFT: Region section (unified ROI + Mask) ======
    regionPanel = uipanel(leftPanel, 'Title', 'Region');
    regionGrid = uigridlayout(regionPanel, [3, 3]);
    regionGrid.ColumnWidth = {'1x', '1x', '1x'};
    regionGrid.RowHeight = {28, 28, 22};
    regionGrid.Padding = [5 2 5 2];
    regionGrid.RowSpacing = 2;

    % Row 1: additive shape buttons
    app.btnDrawRect = uibutton(regionGrid, 'Text', 'Draw Rect', ...
        'ButtonPushedFcn', @onDrawRect);
    app.btnDrawPoly = uibutton(regionGrid, 'Text', 'Draw Poly', ...
        'ButtonPushedFcn', @onDrawPolygon);
    app.btnAddCircle = uibutton(regionGrid, 'Text', 'Add Circle', ...
        'ButtonPushedFcn', @onAddCircle);

    % Row 2: import, cut, clear
    app.btnImportMasks = uibutton(regionGrid, 'Text', 'Import Masks', ...
        'ButtonPushedFcn', @onImportMasks);
    app.btnCutPolygon = uibutton(regionGrid, 'Text', 'Cut Polygon', ...
        'ButtonPushedFcn', @onCutPolygon);
    app.btnClearRegion = uibutton(regionGrid, 'Text', 'Clear', ...
        'ButtonPushedFcn', @onClearRegion);

    app.lblRegionStatus = uilabel(regionGrid, 'Text', 'No region defined');
    app.lblRegionStatus.Layout.Column = [1 3];

    %% ====== LEFT: Parameters section ======
    paramPanel = uipanel(leftPanel, 'Title', 'Parameters');
    paramGrid = uigridlayout(paramPanel, [12, 2]);
    paramGrid.ColumnWidth = {'1x', '1x'};
    paramGrid.RowHeight = {22, 22, 22, 22, 22, 22, 22, 22, 25, 0, 0, 0};
    paramGrid.Padding = [5 2 5 2];
    paramGrid.RowSpacing = 2;

    % Row 1: winsize (auto-rounds to nearest even)
    uilabel(paramGrid, 'Text', 'Subset Size');
    app.edtWinsize = uieditfield(paramGrid, 'numeric', 'Value', 40, ...
        'Tooltip', 'Correlation window size in pixels. Auto-rounds to nearest even number. (winsize)', ...
        'ValueChangedFcn', @onSubsetSizeChanged);

    % Row 2: winstepsize (auto-rounds to nearest power of 2)
    uilabel(paramGrid, 'Text', 'Subset Step');
    app.edtWinstep = uieditfield(paramGrid, 'numeric', 'Value', 16, ...
        'Tooltip', 'Distance between mesh nodes in pixels. Auto-rounds to nearest power of 2. (winstepsize)', ...
        'ValueChangedFcn', @onSubsetStepChanged);

    % Row 3: winsizeMin (auto-rounds to nearest power of 2)
    uilabel(paramGrid, 'Text', 'Min Elem Size');
    app.edtWinsizeMin = uieditfield(paramGrid, 'numeric', 'Value', 8, ...
        'Tooltip', 'Minimum element size for quadtree refinement. Auto-rounds to nearest power of 2. (winsizeMin)', ...
        'ValueChangedFcn', @onMinElemSizeChanged);

    % Row 4: FFTSearchRegion
    uilabel(paramGrid, 'Text', 'FFT Search');
    app.edtFFTSearchRgn = uieditfield(paramGrid, 'numeric', 'Value', 10, ...
        'Tooltip', 'FFT cross-correlation search region in pixels. Auto-scaled if too large. (SizeOfFFTSearchRegion)');

    % Row 5: NewFFTSearch (checkbox)
    uilabel(paramGrid, 'Text', 'Redo FFT');
    app.cbRedoFFT = uicheckbox(paramGrid, 'Text', '', 'Value', true, ...
        'Tooltip', 'Redo FFT cross-correlation search each frame. (NewFFTSearch)');

    % Row 6: referenceMode
    uilabel(paramGrid, 'Text', 'Ref Mode');
    app.ddRefMode = uidropdown(paramGrid, 'Items', {'incremental', 'accumulative'}, ...
        'Value', 'incremental', ...
        'Tooltip', 'incremental=each frame vs previous, accumulative=always vs frame 1. (referenceMode)');

    % Row 7: StrainPlaneFitRad
    uilabel(paramGrid, 'Text', 'Strain Radius');
    app.edtPlaneFitRad = uieditfield(paramGrid, 'numeric', 'Value', 20, ...
        'Tooltip', 'Search radius in pixels for plane-fit strain computation. (StrainPlaneFitRad)');

    % Row 8: Compute Strain checkbox
    uilabel(paramGrid, 'Text', 'Compute Strain');
    app.cbComputeStrain = uicheckbox(paramGrid, 'Text', '', 'Value', true, ...
        'Tooltip', 'Compute strain fields after displacement. Uncheck for displacement-only run.');

    % Row 9: Advanced toggle button
    app.btnAdvanced = uibutton(paramGrid, 'Text', [char(9654), ' Advanced'], ...
        'ButtonPushedFcn', @onToggleAdvanced);
    app.btnAdvanced.Layout.Column = [1 2];

    % Row 10: mu (advanced, hidden by default)
    app.lblAdvMu = uilabel(paramGrid, 'Text', 'Penalty mu', 'Visible', 'off');
    app.edtMu = uieditfield(paramGrid, 'numeric', 'Value', 1e-3, ...
        'ValueDisplayFormat', '%.1e', 'Visible', 'off', ...
        'Tooltip', 'ADMM augmented Lagrangian penalty weight. (mu)');

    % Row 11: alpha (advanced, hidden by default)
    app.lblAdvAlpha = uilabel(paramGrid, 'Text', 'Regularize alpha', 'Visible', 'off');
    app.edtAlpha = uieditfield(paramGrid, 'numeric', 'Value', 0, ...
        'Visible', 'off', ...
        'Tooltip', 'Regularization parameter in global subproblem. (alpha)');

    % Row 12: ADMM_maxIter (advanced, hidden by default)
    app.lblAdvADMM = uilabel(paramGrid, 'Text', 'ADMM Iters', 'Visible', 'off');
    app.edtADMM = uieditfield(paramGrid, 'numeric', 'Value', 3, ...
        'Visible', 'off', ...
        'Tooltip', 'Maximum number of outer ADMM iterations. (ADMM_maxIter)');

    % (ROI section removed — merged into Region panel above)

    %% ====== LEFT: Run / Stop buttons ======
    app.btnRun = uibutton(leftPanel, 'Text', 'Run DIC', ...
        'BackgroundColor', [0.2 0.7 0.3], 'FontColor', 'w', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @onRun);
    app.btnStop = uibutton(leftPanel, 'Text', 'Stop', ...
        'BackgroundColor', [0.8 0.2 0.2], 'FontColor', 'w', 'FontWeight', 'bold', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @onStop);

    %% ====== LEFT: Save Results ======
    app.btnSave = uibutton(leftPanel, 'Text', 'Save Results', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @onSave);

    %% ====== LEFT: Compute Strain (post-hoc) ======
    app.btnComputeStrain = uibutton(leftPanel, 'Text', 'Compute Strain', ...
        'Enable', 'off', ...
        'Tooltip', 'Compute strain on existing displacement results', ...
        'ButtonPushedFcn', @onComputeStrain);

    %% ====== RIGHT: Main axes ======
    app.axMain = uiaxes(rightPanel);
    app.axMain.Title.String = 'Load images to begin';
    app.axMain.XLabel.String = 'x (pixels)';
    app.axMain.YLabel.String = 'y (pixels)';

    %% ====== RIGHT: Frame/Field controls ======
    ctrlGrid = uigridlayout(rightPanel, [1, 6]);
    ctrlGrid.ColumnWidth = {60, 30, '1x', 30, 60, '1x'};
    ctrlGrid.Padding = [5 0 5 0];

    uilabel(ctrlGrid, 'Text', 'Frame:');
    app.btnPrevFrame = uibutton(ctrlGrid, 'Text', '<', ...
        'Enable', 'off', 'ButtonPushedFcn', @(~,~) changeFrame(-1));
    app.sldFrame = uislider(ctrlGrid, 'Limits', [1 2], 'Value', 1, ...
        'Enable', 'off', 'ValueChangedFcn', @onFrameChanged);
    app.btnNextFrame = uibutton(ctrlGrid, 'Text', '>', ...
        'Enable', 'off', 'ButtonPushedFcn', @(~,~) changeFrame(1));

    uilabel(ctrlGrid, 'Text', 'Field:');
    app.ddField = uidropdown(ctrlGrid, ...
        'Items', {'Disp U','Disp V','exx','exy','eyy','vonMises','maxshear', ...
                  'principal max','principal min'}, ...
        'Value', 'Disp U', 'Enable', 'off', ...
        'ValueChangedFcn', @onFieldChanged);

    %% ====== RIGHT: Color Range + Overlay controls (2 rows) ======
    climGrid = uigridlayout(rightPanel, [2, 6]);
    climGrid.ColumnWidth = {80, 55, 60, 80, 60, '1x'};
    climGrid.RowHeight = {25, 25};
    climGrid.Padding = [5 0 5 0];
    climGrid.RowSpacing = 2;

    % Row 1: Color range
    uilabel(climGrid, 'Text', 'Color Range:');
    app.cbAutoClim = uicheckbox(climGrid, 'Text', 'Auto', 'Value', true, ...
        'ValueChangedFcn', @onAutoClimChanged);
    uilabel(climGrid, 'Text', 'Min:', 'HorizontalAlignment', 'right');
    app.efClimMin = uieditfield(climGrid, 'numeric', 'Value', 0, ...
        'Enable', 'off', 'ValueChangedFcn', @(~,~) updateDisplay());
    uilabel(climGrid, 'Text', 'Max:', 'HorizontalAlignment', 'right');
    app.efClimMax = uieditfield(climGrid, 'numeric', 'Value', 1, ...
        'Enable', 'off', 'ValueChangedFcn', @(~,~) updateDisplay());

    % Row 2: Colormap, Show Image, Alpha
    uilabel(climGrid, 'Text', 'Colormap:');
    app.ddColormap = uidropdown(climGrid, ...
        'Items', {'jet','parula','hot','turbo','gray','coolwarm','RdYlBu'}, ...
        'Value', 'jet', 'ValueChangedFcn', @(~,~) updateDisplay());
    app.cbShowImage = uicheckbox(climGrid, 'Text', 'Show Image', 'Value', true, ...
        'ValueChangedFcn', @(~,~) updateDisplay());
    uilabel(climGrid, 'Text', 'Alpha:', 'HorizontalAlignment', 'right');
    app.efAlpha = uieditfield(climGrid, 'numeric', 'Value', 70, ...
        'Limits', [0 100], 'ValueChangedFcn', @(~,~) updateDisplay(), ...
        'Tooltip', 'Overlay transparency 0-100%');
    uilabel(climGrid, 'Text', '%');

    %% ====== RIGHT: Progress + Log ======
    progGrid = uigridlayout(rightPanel, [2, 1]);
    progGrid.RowHeight = {24, '1x'};
    progGrid.Padding = [5 5 5 5];
    progGrid.RowSpacing = 5;

    % Custom progress bar: outer panel with 3-col grid (fill | space | label)
    app.progOuter = uipanel(progGrid, 'BorderType', 'line', ...
        'BackgroundColor', [0.92 0.92 0.92]);
    app.progOuterGrid = uigridlayout(app.progOuter, [1, 3]);
    app.progOuterGrid.ColumnWidth = {0, '1x', 35};
    app.progOuterGrid.Padding = [0 0 0 0];
    app.progOuterGrid.ColumnSpacing = 0;
    app.progBar = uipanel(app.progOuterGrid, 'BorderType', 'none', ...
        'BackgroundColor', [0.2 0.6 1]);
    uilabel(app.progOuterGrid, 'Text', '');  % spacer (gray shows through)
    app.progLabel = uilabel(app.progOuterGrid, 'Text', '0%', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontColor', [0.3 0.3 0.3]);

    app.txtLog = uitextarea(progGrid, 'Value', {'Ready.'}, 'Editable', 'off');

    %% ======================================================
    %  Callback functions
    % =======================================================

    function onClose(~, ~)
        delete(app.fig);
    end

    function onLoadImages(~, ~)
        folder = uigetdir(app.lastImgDir, 'Select images folder');
        if folder == 0, figure(app.fig); return; end
        figure(app.fig);  % restore focus after system dialog
        app.lastImgDir = folder;
        app.imgFolder = folder;

        % Find image files
        exts = {'*.jpg','*.jpeg','*.tif','*.tiff','*.bmp','*.png','*.jp2'};
        allFiles = [];
        for k = 1:length(exts)
            allFiles = [allFiles; dir(fullfile(folder, exts{k}))]; %#ok<AGROW>
        end
        if isempty(allFiles)
            uialert(app.fig, 'No image files found in selected folder.', 'Error');
            return;
        end

        % Sort by name
        [~, sortIdx] = sort({allFiles.name});
        allFiles = allFiles(sortIdx);

        % Build file_name cell array (same format as read_images)
        nFiles = length(allFiles);
        app.file_name = cell(6, nFiles);
        for k = 1:nFiles
            app.file_name{1,k} = allFiles(k).name;
            app.file_name{2,k} = allFiles(k).folder;
        end

        % Load images
        app.Img = cell(1, nFiles);
        for k = 1:nFiles
            fpath = fullfile(allFiles(k).folder, allFiles(k).name);
            img = imread(fpath);
            [~, ~, nc] = size(img);
            if nc == 3, img = rgb2gray(img);
            elseif nc == 4, img = rgb2gray(img(:,:,1:3));
            end
            app.Img{k} = double(img)';
        end

        % Update listbox
        app.lstImages.Items = {allFiles.name};
        app.lstImages.Value = allFiles(1).name;
        logMsg(sprintf('Loaded %d images from %s', nFiles, folder));

        % Show first image
        showImage(1);
    end

    function onImageSelected(~, ~)
        idx = find(strcmp(app.lstImages.Items, app.lstImages.Value));
        if ~isempty(idx)
            showImage(idx);
        end
    end

    function showImage(idx)
        if idx > 0 && idx <= length(app.Img)
            cla(app.axMain);
            imgDisp = app.Img{idx}';  % transpose code-space → display orientation
            [imgH, imgW] = size(imgDisp);
            imgRGB = repmat(mat2gray(imgDisp), [1, 1, 3]);
            image(app.axMain, 'XData', [1, imgW], 'YData', [imgH, 1], 'CData', imgRGB);
            set(app.axMain, 'YDir', 'normal');
            axis(app.axMain, 'equal');
            xlim(app.axMain, [1, imgW]);
            ylim(app.axMain, [1, imgH]);
            app.axMain.Title.String = sprintf('Image %d: %s', idx, app.file_name{1,idx});
            app.axMain.XLabel.String = 'x (pixels)';
            app.axMain.YLabel.String = 'y (pixels)';
        end
    end

    %% ====== Region callbacks (unified ROI/Mask) ======

    function onDrawRect(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        % Show existing mask overlay so user sees current regions
        if ~isempty(app.regionOps) && ~isempty(app.ImgMask)
            showMaskOverlay();
        else
            showImage(1);
        end
        app.axMain.Title.String = 'Draw rectangle ROI, then double-click to confirm';
        roi = drawrectangle(app.axMain, 'Label', 'ROI');
        if isempty(roi.Position), return; end
        app.regionOps{end+1} = struct('type','rect', 'data',roi.Position);
        recomputeMask();
        updateRegionInfo();
    end

    function onDrawPolygon(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        if ~isempty(app.regionOps) && ~isempty(app.ImgMask)
            showMaskOverlay();
        else
            showImage(1);
        end
        app.axMain.Title.String = 'Draw polygon ROI, then double-click to close';
        roi = drawpolygon(app.axMain, 'Label', 'ROI');
        if isempty(roi.Position), return; end
        app.regionOps{end+1} = struct('type','polygon', 'data',roi.Position);
        recomputeMask();
        updateRegionInfo();
    end

    function onAddCircle(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        if ~isempty(app.regionOps) && ~isempty(app.ImgMask)
            showMaskOverlay();
        else
            showImage(1);
        end
        app.axMain.Title.String = 'Draw circle ROI, then confirm';
        roi = drawcircle(app.axMain, 'Label', 'Circle');
        if roi.Radius <= 0, return; end
        app.regionOps{end+1} = struct('type','circle', ...
            'data',[roi.Center(1), roi.Center(2), roi.Radius]);
        recomputeMask();
        updateRegionInfo();
    end

    function onCutPolygon(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        if isempty(app.regionOps)
            uialert(app.fig, 'Define a region first before cutting.', 'Cut Polygon');
            return;
        end
        showMaskOverlay();
        app.axMain.Title.String = 'Draw cut polygon, then double-click to close';
        roi = drawpolygon(app.axMain, 'Label', 'Cut', 'Color', 'r');
        if isempty(roi.Position), return; end
        app.regionOps{end+1} = struct('type','cut_polygon', 'data',roi.Position);
        recomputeMask();
        updateRegionInfo();
    end

    function onImportMasks(~, ~)
        folder = uigetdir(app.lastMaskDir, 'Select mask files folder');
        if folder == 0, figure(app.fig); return; end
        figure(app.fig);  % restore focus after system dialog
        app.lastMaskDir = folder;
        app.maskFolder = folder;

        exts = {'*.jpg','*.jpeg','*.tif','*.tiff','*.bmp','*.png','*.jp2'};
        allFiles = [];
        for k = 1:length(exts)
            allFiles = [allFiles; dir(fullfile(folder, exts{k}))]; %#ok<AGROW>
        end
        if isempty(allFiles)
            uialert(app.fig, 'No mask files found in selected folder.', 'Error');
            return;
        end

        [~, sortIdx] = sort({allFiles.name});
        allFiles = allFiles(sortIdx);

        nFiles = length(allFiles);
        app.mask_file_name = cell(6, nFiles);
        importedMasks = cell(1, nFiles);
        for k = 1:nFiles
            fpath = fullfile(allFiles(k).folder, allFiles(k).name);
            app.mask_file_name{1,k} = allFiles(k).name;
            app.mask_file_name{2,k} = allFiles(k).folder;
            img = imread(fpath);
            [~, ~, nc] = size(img);
            if nc == 3, img = rgb2gray(img); end
            importedMasks{k} = logical(img)';
        end

        % Use {cellArray} syntax to prevent MATLAB struct expansion
        app.regionOps{end+1} = struct('type','imported', 'data',{importedMasks});
        recomputeMask();
        updateRegionInfo();
        logMsg(sprintf('Imported %d masks from %s', nFiles, folder));
    end

    function onClearRegion(~, ~)
        app.regionOps = {};
        app.ImgMask = {};
        app.roiGridx = [];
        app.roiGridy = [];
        updateRegionInfo();
        if ~isempty(app.Img), showImage(1); end
    end

    function recomputeMask()
        if isempty(app.regionOps)
            app.ImgMask = {};
            app.roiGridx = [];
            app.roiGridy = [];
            return;
        end

        imgSize = size(app.Img{1}); % [originalCols, originalRows] (transposed)
        dispRows = imgSize(2);
        dispCols = imgSize(1);
        nFrames = length(app.Img);

        % Check if any 'imported' ops exist (per-frame masks)
        hasImported = false;
        for k = 1:length(app.regionOps)
            if strcmp(app.regionOps{k}.type, 'imported')
                hasImported = true; break;
            end
        end

        if ~hasImported
            % No imports: compute mask once, replicate to all frames
            mask = applyOps(false(dispRows, dispCols), app.regionOps, 1, dispRows, dispCols);
            codeMask = logical(mask');
            app.ImgMask = repmat({codeMask}, 1, nFrames);
        else
            % Per-frame computation (imported masks may differ per frame)
            app.ImgMask = cell(1, nFrames);
            for f = 1:nFrames
                mask = applyOps(false(dispRows, dispCols), app.regionOps, f, dispRows, dispCols);
                app.ImgMask{f} = logical(mask');
            end
        end

        % Compute ROI bounding box from first frame mask
        firstMask = app.ImgMask{1};
        [d1idx, d2idx] = find(firstMask);
        if ~isempty(d1idx)
            app.roiGridx = [min(d1idx), max(d1idx)];
            app.roiGridy = [min(d2idx), max(d2idx)];
        else
            app.roiGridx = [];
            app.roiGridy = [];
        end

        showMaskOverlay();
    end

    function mask = applyOps(mask, ops, frameIdx, dispRows, dispCols)
        % Apply operations in order to produce the final display-space mask.
        for k = 1:length(ops)
            op = ops{k};
            switch op.type
                case 'rect'
                    pos = op.data;  % [x, y, w, h]
                    r1 = max(1, round(pos(2)));
                    r2 = min(dispRows, round(pos(2)+pos(4)));
                    c1 = max(1, round(pos(1)));
                    c2 = min(dispCols, round(pos(1)+pos(3)));
                    rectMask = false(dispRows, dispCols);
                    rectMask(r1:r2, c1:c2) = true;
                    mask = mask | rectMask;

                case 'polygon'
                    polyMask = poly2mask(op.data(:,1), op.data(:,2), dispRows, dispCols);
                    mask = mask | polyMask;

                case 'circle'
                    cx = op.data(1); cy = op.data(2); r = op.data(3);
                    [cc, rr] = meshgrid(1:dispCols, 1:dispRows);
                    circleMask = ((cc - cx).^2 + (rr - cy).^2) <= r^2;
                    mask = mask | circleMask;

                case 'cut_polygon'
                    cutMask = poly2mask(op.data(:,1), op.data(:,2), dispRows, dispCols);
                    mask = mask & ~cutMask;

                case 'imported'
                    importedMasks = op.data;
                    idx = min(frameIdx, length(importedMasks));
                    % Imported masks are already in code space — transpose to display
                    impDispMask = importedMasks{idx}';
                    mask = mask | impDispMask;
            end
        end
    end

    function showMaskOverlay()
        if isempty(app.Img) || isempty(app.ImgMask), return; end
        cla(app.axMain);
        img = app.Img{1}';   % display orientation
        mask = app.ImgMask{1}'; % display orientation
        imshow(img, [], 'Parent', app.axMain);
        hold(app.axMain, 'on');
        % Semi-transparent red overlay on selected ROI region
        redOverlay = cat(3, ones(size(mask)), zeros(size(mask)), zeros(size(mask)));
        h = imshow(redOverlay, 'Parent', app.axMain);
        set(h, 'AlphaData', 0.3 * double(mask));
        hold(app.axMain, 'off');
        nOps = length(app.regionOps);
        app.axMain.Title.String = sprintf('Region: %d operation(s)', nOps);
    end

    function updateRegionInfo()
        if isempty(app.regionOps)
            app.lblRegionStatus.Text = 'No region defined';
            return;
        end

        % Count operations by type
        types = cellfun(@(op) op.type, app.regionOps, 'UniformOutput', false);
        parts = {};
        typeNames = {'rect','polygon','circle','cut_polygon','imported'};
        displayNames = {'rect','poly','circle','cut','imported'};
        for k = 1:length(typeNames)
            n = sum(strcmp(types, typeNames{k}));
            if n > 0
                parts{end+1} = sprintf('%d %s', n, displayNames{k}); %#ok<AGROW>
            end
        end
        app.lblRegionStatus.Text = strjoin(parts, ' + ');
    end

    %% ====== Parameter auto-rounding callbacks ======

    function onSubsetSizeChanged(~, ~)
        val = app.edtWinsize.Value;
        val = max(2, 2 * round(val / 2));  % nearest even, min 2
        app.edtWinsize.Value = val;
    end

    function onSubsetStepChanged(~, ~)
        val = app.edtWinstep.Value;
        val = max(2, 2^round(log2(max(1, val))));  % nearest power of 2, min 2
        app.edtWinstep.Value = val;
    end

    function onMinElemSizeChanged(~, ~)
        val = app.edtWinsizeMin.Value;
        val = max(1, 2^round(log2(max(1, val))));  % nearest power of 2, min 1
        app.edtWinsizeMin.Value = val;
    end

    function onToggleAdvanced(~, ~)
        app.advancedVisible = ~app.advancedVisible;
        if app.advancedVisible
            vis = 'on';
            paramGrid.RowHeight(10:12) = {22, 22, 22};
            app.btnAdvanced.Text = [char(9660), ' Advanced'];
        else
            vis = 'off';
            paramGrid.RowHeight(10:12) = {0, 0, 0};
            app.btnAdvanced.Text = [char(9654), ' Advanced'];
        end
        app.lblAdvMu.Visible = vis;
        app.edtMu.Visible = vis;
        app.lblAdvAlpha.Visible = vis;
        app.edtAlpha.Visible = vis;
        app.lblAdvADMM.Visible = vis;
        app.edtADMM.Visible = vis;
    end

    %% ====== Run/Stop callbacks ======

    function onRun(~, ~)
        % Validate inputs
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        if isempty(app.regionOps)
            uialert(app.fig, 'Define a region first (draw ROI or import masks).', 'Error'); return;
        end
        if isempty(app.ImgMask) || isempty(app.roiGridx)
            uialert(app.fig, 'Region mask is empty. Redefine the region.', 'Error'); return;
        end

        % Build DICpara from GUI fields
        DICpara = struct();
        DICpara.winsize = app.edtWinsize.Value;
        DICpara.winstepsize = app.edtWinstep.Value;
        DICpara.winsizeMin = app.edtWinsizeMin.Value;
        DICpara.SizeOfFFTSearchRegion = app.edtFFTSearchRgn.Value;
        DICpara.NewFFTSearch = double(app.cbRedoFFT.Value);
        DICpara.mu = app.edtMu.Value;
        DICpara.alpha = app.edtAlpha.Value;
        DICpara.ADMM_maxIter = app.edtADMM.Value;
        DICpara.referenceMode = app.ddRefMode.Value;
        DICpara.StrainPlaneFitRad = app.edtPlaneFitRad.Value;
        DICpara.gridxyROIRange.gridx = app.roiGridx;
        DICpara.gridxyROIRange.gridy = app.roiGridy;
        DICpara.LoadImgMethod = 0;
        DICpara.imgFolder = app.imgFolder;
        DICpara.showPlots = false;  % GUI controls all visualization

        % Detect bit depth
        try
            imgInfo = imfinfo(fullfile(app.file_name{2,1}, app.file_name{1,1}));
            DICpara.imgBitDepth = imgInfo.BitDepth;
        catch
            DICpara.imgBitDepth = 8;
        end
        DICpara.ImgSize = size(app.Img{1});

        % Merge with defaults
        DICpara = dicpara_default(DICpara);

        % UI state
        app.stopRequested = false;
        app.isRunning = true;
        app.btnRun.Enable = 'off';
        app.btnStop.Enable = 'on';
        setProgress(0);
        logMsg('Starting DIC computation...');

        % Run pipeline
        try
            app.results = run_aldic(DICpara, app.file_name, app.Img, app.ImgMask, ...
                'ProgressFcn', @guiProgress, ...
                'StopFcn', @() app.stopRequested, ...
                'ComputeStrain', app.cbComputeStrain.Value);
            logMsg('Computation complete!');
            enableResultsControls();
        catch ME
            logMsg(sprintf('ERROR: %s', ME.message));
            uialert(app.fig, ME.message, 'Computation Error');
        end

        % Restore UI state
        app.isRunning = false;
        app.btnRun.Enable = 'on';
        app.btnStop.Enable = 'off';
    end

    function onStop(~, ~)
        app.stopRequested = true;
        logMsg('Stop requested. Finishing current frame...');
    end

    function guiProgress(frac, msg)
        setProgress(frac);
        logMsg(msg);
        drawnow;
    end

    function logMsg(msg)
        currentLog = app.txtLog.Value;
        timestamp = datestr(now, 'HH:MM:SS');
        newLine = sprintf('[%s] %s', timestamp, msg);
        app.txtLog.Value = [currentLog; {newLine}];
        app.txtLog.scroll('bottom');
    end

    function setProgress(frac)
        % Update custom progress bar (frac: 0.0 to 1.0)
        pct = max(0, min(100, round(frac * 100)));
        % Resize filled portion via grid column widths
        if pct == 0
            app.progOuterGrid.ColumnWidth = {0, '1x', 35};
        elseif pct >= 100
            app.progOuterGrid.ColumnWidth = {'1x', 0, 35};
        else
            app.progOuterGrid.ColumnWidth = {sprintf('%dx', pct), sprintf('%dx', 100-pct), 35};
        end
        app.progLabel.Text = sprintf('%d%%', pct);
    end

    function enableResultsControls()
        nResults = length(app.results.ResultStrain);
        if nResults < 1, return; end

        app.sldFrame.Limits = [1 max(nResults, 1.01)];
        app.sldFrame.Value = 1;
        app.sldFrame.Enable = 'on';
        if nResults > 1
            app.sldFrame.MajorTicks = 1:nResults;
        end
        app.btnPrevFrame.Enable = 'on';
        app.btnNextFrame.Enable = 'on';
        app.ddField.Enable = 'on';
        app.btnSave.Enable = 'on';

        % Detect whether strain was computed
        hasStrain = isfield(app.results.ResultStrain{1}, 'strain_exx');
        if hasStrain
            app.ddField.Items = {'Disp U','Disp V','exx','exy','eyy', ...
                'vonMises','maxshear','principal max','principal min'};
            app.btnComputeStrain.Enable = 'off';
        else
            app.ddField.Items = {'Disp U','Disp V'};
            app.btnComputeStrain.Enable = 'on';
        end
        % Reset to Disp U when items change
        app.ddField.Value = 'Disp U';

        updateDisplay();
    end

    function changeFrame(delta)
        newVal = round(app.sldFrame.Value) + delta;
        newVal = max(1, min(newVal, app.sldFrame.Limits(2)));
        app.sldFrame.Value = newVal;
        updateDisplay();
    end

    function onFrameChanged(~, ~)
        updateDisplay();
    end

    function onFieldChanged(~, ~)
        % Reset to auto when field changes so user sees the natural range
        app.cbAutoClim.Value = true;
        app.efClimMin.Enable = 'off';
        app.efClimMax.Enable = 'off';
        updateDisplay();
    end

    function onAutoClimChanged(~, ~)
        if app.cbAutoClim.Value
            app.efClimMin.Enable = 'off';
            app.efClimMax.Enable = 'off';
        else
            app.efClimMin.Enable = 'on';
            app.efClimMax.Enable = 'on';
        end
        updateDisplay();
    end

    function onSave(~, ~)
        if isempty(app.results)
            uialert(app.fig, 'No results to save.', 'Save');
            return;
        end

        % Build default filename
        if ~isempty(app.file_name) && size(app.file_name,2) >= 1
            [~, imgname, ~] = fileparts(app.file_name{1,end});
        else
            imgname = 'unknown';
        end
        ws = app.results.DICpara.winsize;
        st = app.results.DICpara.winstepsize;
        defaultName = sprintf('results_%s_ws%d_st%d.mat', imgname, ws, st);

        [fname, fpath] = uiputfile('*.mat', 'Save Results As', defaultName);
        if fname == 0, return; end  % user cancelled

        results = app.results; %#ok<NASGU>
        save(fullfile(fpath, fname), '-struct', 'results');
        logMsg(sprintf('Results saved to %s', fullfile(fpath, fname)));
    end

    function onComputeStrain(~, ~)
        % Post-hoc strain computation on existing displacement results
        if isempty(app.results)
            uialert(app.fig, 'Run DIC first to get displacement results.', 'Error');
            return;
        end

        % Check if strain already computed
        RS1 = app.results.ResultStrain{1};
        if isfield(RS1, 'strain_exx')
            uialert(app.fig, 'Strain fields already computed.', 'Info');
            return;
        end

        DICpara = app.results.DICpara;
        DICmesh = app.results.DICmesh;

        app.btnComputeStrain.Enable = 'off';
        setProgress(0);
        logMsg('Computing strain fields from existing displacements...');

        try
            % Normalize images (same as run_aldic Section 2b)
            [ImgNormalized, ~] = normalize_img(app.Img, DICpara.gridxyROIRange);
            nFrames = length(ImgNormalized);

            coordinatesFEM = app.results.ResultFEMeshEachFrame{1}.coordinatesFEM;
            elementsFEM = app.results.ResultFEMeshEachFrame{1}.elementsFEM;
            DICmesh.coordinatesFEM = coordinatesFEM;
            DICmesh.elementsFEM = elementsFEM;
            coordinatesFEMWorld = DICpara.um2px * [coordinatesFEM(:,1), ...
                size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)];

            Rad = [];
            if DICpara.MethodToComputeStrain == 2
                Rad = DICpara.StrainPlaneFitRad;
            end
            if DICpara.smoothness > 0
                DICpara.skipExtraSmoothing = 0;
            else
                DICpara.skipExtraSmoothing = 1;
            end

            for ImgSeqNum = 2:nFrames
                setProgress((ImgSeqNum-2)/(nFrames-1) * 0.9);
                logMsg(sprintf('Computing strain for frame %d/%d', ImgSeqNum, nFrames));
                drawnow;

                % Get accumulated displacement
                ULocal = app.results.ResultDisp{ImgSeqNum-1}.U_accum;
                UWorld = DICpara.um2px * ULocal;
                UWorld(2:2:end) = -UWorld(2:2:end);

                % Image gradients for reference and deformed
                if strcmp(DICpara.referenceMode, 'accumulative')
                    fNormalizedMask = double(app.ImgMask{1});
                    fNormalized = ImgNormalized{1} .* fNormalizedMask;
                else
                    fNormalizedMask = double(app.ImgMask{ImgSeqNum-1});
                    fNormalized = ImgNormalized{ImgSeqNum-1} .* fNormalizedMask;
                end
                Df = img_gradient(fNormalized, fNormalized, fNormalizedMask);

                gNormalizedMask = double(app.ImgMask{ImgSeqNum});
                gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask;
                Dg = img_gradient(gNormalized, gNormalized, gNormalizedMask);
                DICpara.ImgRefMask = fNormalizedMask;

                % Smooth displacements
                SmoothTimes = 0;
                while DICpara.skipExtraSmoothing == 0 && SmoothTimes < 3
                    ULocal = smooth_disp_rbf(ULocal, DICmesh, DICpara);
                    SmoothTimes = SmoothTimes + 1;
                end

                % Compute strain field
                [FStraintemp, FStrainWorld] = compute_strain(ULocal, [], coordinatesFEM, ...
                    DICmesh, DICpara, Df, Dg, Rad);

                % Extract strain components
                u_x = FStrainWorld(1:4:end); v_x = FStrainWorld(2:4:end);
                u_y = FStrainWorld(3:4:end); v_y = FStrainWorld(4:4:end);
                strain_exx = u_x;
                strain_exy = 0.5*(v_x + u_y);
                strain_eyy = v_y;
                strain_maxshear = sqrt((0.5*(strain_exx-strain_eyy)).^2 + strain_exy.^2);
                strain_principal_max = 0.5*(strain_exx+strain_eyy) + strain_maxshear;
                strain_principal_min = 0.5*(strain_exx+strain_eyy) - strain_maxshear;
                strain_vonMises = sqrt(strain_principal_max.^2 + strain_principal_min.^2 - ...
                    strain_principal_max.*strain_principal_min + 3*strain_maxshear.^2);

                % Update ResultStrain with full data
                app.results.ResultStrain{ImgSeqNum-1} = struct( ...
                    'strainxCoord', coordinatesFEMWorld(:,1), ...
                    'strainyCoord', coordinatesFEMWorld(:,2), ...
                    'dispu', UWorld(1:2:end), 'dispv', UWorld(2:2:end), ...
                    'dudx', FStraintemp(1:4:end), 'dvdx', FStraintemp(2:4:end), ...
                    'dudy', FStraintemp(3:4:end), 'dvdy', FStraintemp(4:4:end), ...
                    'strain_exx', strain_exx, 'strain_exy', strain_exy, ...
                    'strain_eyy', strain_eyy, ...
                    'strain_principal_max', strain_principal_max, ...
                    'strain_principal_min', strain_principal_min, ...
                    'strain_maxshear', strain_maxshear, ...
                    'strain_vonMises', strain_vonMises);
            end

            setProgress(1);
            logMsg('Strain computation complete!');
            enableResultsControls();
        catch ME
            logMsg(sprintf('ERROR: %s', ME.message));
            uialert(app.fig, ME.message, 'Strain Computation Error');
        end
        app.btnComputeStrain.Enable = 'on';
    end

    function updateDisplay()
        if isempty(app.results), return; end

        frameIdx = round(app.sldFrame.Value);
        fieldName = app.ddField.Value;

        if frameIdx < 1 || frameIdx > length(app.results.ResultStrain)
            return;
        end

        RS = app.results.ResultStrain{frameIdx};
        if isempty(RS), return; end

        % Select data field (gracefully handle missing strain fields)
        hasStrain = isfield(RS, 'strain_exx');
        switch fieldName
            case 'Disp U',          data = RS.dispu;
            case 'Disp V',          data = RS.dispv;
            case 'exx',             if hasStrain, data = RS.strain_exx; else, data = RS.dispu; end
            case 'exy',             if hasStrain, data = RS.strain_exy; else, data = RS.dispu; end
            case 'eyy',             if hasStrain, data = RS.strain_eyy; else, data = RS.dispu; end
            case 'vonMises',        if hasStrain, data = RS.strain_vonMises; else, data = RS.dispu; end
            case 'maxshear',        if hasStrain, data = RS.strain_maxshear; else, data = RS.dispu; end
            case 'principal max',   if hasStrain, data = RS.strain_principal_max; else, data = RS.dispu; end
            case 'principal min',   if hasStrain, data = RS.strain_principal_min; else, data = RS.dispu; end
            otherwise,              data = RS.dispu;
        end

        coordWorld = [RS.strainxCoord, RS.strainyCoord];
        elemFEM = app.results.ResultFEMeshEachFrame{1}.elementsFEM;
        um2px = app.results.DICpara.um2px;

        cla(app.axMain);

        % --- Background image (grayscale, rendered as RGB truecolor) ---
        if app.cbShowImage.Value && ~isempty(app.Img)
            imgIdx = min(frameIdx + 1, length(app.Img));  % ResultStrain{k} → Img{k+1}
            imgCode = app.Img{imgIdx};              % code-space (transposed)
            imgDisp = imgCode';                     % display orientation [H x W]
            [imgH, imgW] = size(imgDisp);

            % Convert to RGB so it doesn't compete with the overlay colormap
            imgRGB = repmat(mat2gray(imgDisp), [1, 1, 3]);

            % World-space extents: x_world = um2px*col, y_world = um2px*(H+1-row)
            xLim = um2px * [1, imgW];
            yLim = um2px * [imgH, 1];  % row 1 → top (high y), last row → bottom (low y)

            image(app.axMain, 'XData', xLim, 'YData', yLim, 'CData', imgRGB);
            hold(app.axMain, 'on');
        end

        % --- Overlay: colored DIC result patch with transparency ---
        faces = elemFEM(:, 1:4);
        vertices = coordWorld;
        faceAlpha = app.efAlpha.Value / 100;

        patch(app.axMain, 'Faces', faces, 'Vertices', vertices, ...
            'FaceVertexCData', data(:), ...
            'FaceColor', 'interp', 'EdgeColor', 'none', ...
            'FaceAlpha', faceAlpha);

        hold(app.axMain, 'off');
        view(app.axMain, 2);
        axis(app.axMain, 'equal');
        set(app.axMain, 'YDir', 'normal');

        % Fit axes to full image if shown, otherwise to patch extent
        if app.cbShowImage.Value && ~isempty(app.Img)
            xlim(app.axMain, xLim);
            ylim(app.axMain, sort(yLim));
        else
            axis(app.axMain, 'tight');
        end

        colorbar(app.axMain);
        colormap(app.axMain, getColormapData(app.ddColormap.Value));

        % Apply color range
        if app.cbAutoClim.Value
            % Percentile-based auto range (2nd to 98th)
            validData = data(~isnan(data));
            if ~isempty(validData)
                sortedData = sort(validData);
                n = length(sortedData);
                climLo = sortedData(max(1, round(0.02*n)));
                climHi = sortedData(min(n, round(0.98*n)));
                if climLo >= climHi
                    climHi = climLo + eps;
                end
                clim(app.axMain, [climLo, climHi]);
                app.efClimMin.Value = climLo;
                app.efClimMax.Value = climHi;
            end
        else
            % User-specified range
            if app.efClimMin.Value < app.efClimMax.Value
                clim(app.axMain, [app.efClimMin.Value, app.efClimMax.Value]);
            end
        end

        app.axMain.Title.String = sprintf('Frame %d - %s', frameIdx, fieldName);
        app.axMain.XLabel.String = 'x (pixels)';
        app.axMain.YLabel.String = 'y (pixels)';
    end

    function cmap = getColormapData(name)
        % Return Nx3 colormap matrix for the given name
        switch name
            case 'coolwarm'
                try cmap = coolwarm(256); catch, cmap = jet(256); end
            case 'RdYlBu'
                try
                    s = load('./plotting/colormap_RdYlBu.mat', 'cMap');
                    cmap = s.cMap;
                catch
                    cmap = jet(256);
                end
            otherwise
                % Built-in MATLAB colormaps: jet, parula, hot, turbo, gray
                try cmap = feval(name, 256); catch, cmap = jet(256); end
        end
    end

end
