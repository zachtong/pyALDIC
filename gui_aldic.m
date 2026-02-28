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

    % Region state (unified ROI/Mask)
    app.regionType = 'none';     % 'none', 'rect', 'polygon', 'imported'
    app.regionRect = [];          % [x,y,w,h] in display coords (rect mode)
    app.regionVertices = [];      % Outer polygon vertices Nx2 display coords
    app.holeVertices = {};        % Cell array of hole polygon vertices
    app.roiGridx = [];
    app.roiGridy = [];
    app.advancedVisible = false;

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
    leftPanel = uigridlayout(mainGrid, [6, 1]);
    leftPanel.RowHeight = {130, 150, '1x', 35, 35, 35};
    leftPanel.Padding = [0 0 0 0];
    leftPanel.RowSpacing = 5;

    % --- Right panel ---
    rightPanel = uigridlayout(mainGrid, [4, 1]);
    rightPanel.RowHeight = {'1x', 35, 30, 160};
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
    regionGrid = uigridlayout(regionPanel, [4, 3]);
    regionGrid.ColumnWidth = {'1x', '1x', '1x'};
    regionGrid.RowHeight = {28, 28, 22, 22};
    regionGrid.Padding = [5 2 5 2];
    regionGrid.RowSpacing = 2;

    app.btnDrawRect = uibutton(regionGrid, 'Text', 'Draw Rect', ...
        'ButtonPushedFcn', @onDrawRect);
    app.btnDrawPoly = uibutton(regionGrid, 'Text', 'Draw Poly', ...
        'ButtonPushedFcn', @onDrawPolygon);
    app.btnImportMasks = uibutton(regionGrid, 'Text', 'Import Masks', ...
        'ButtonPushedFcn', @onImportMasks);

    app.btnAddHole = uibutton(regionGrid, 'Text', 'Add Hole', ...
        'Enable', 'off', 'ButtonPushedFcn', @onAddHole);
    app.btnClearRegion = uibutton(regionGrid, 'Text', 'Clear', ...
        'ButtonPushedFcn', @onClearRegion);
    uilabel(regionGrid, 'Text', '');  % empty slot

    app.lblRegionStatus = uilabel(regionGrid, 'Text', 'No region defined');
    app.lblRegionStatus.Layout.Column = [1 3];

    app.lblROIInfo = uilabel(regionGrid, 'Text', 'ROI: [ - ]');
    app.lblROIInfo.Layout.Column = [1 3];

    %% ====== LEFT: Parameters section ======
    paramPanel = uipanel(leftPanel, 'Title', 'Parameters');
    paramGrid = uigridlayout(paramPanel, [11, 2]);
    paramGrid.ColumnWidth = {'1x', '1x'};
    paramGrid.RowHeight = {22, 22, 22, 22, 22, 22, 22, 25, 0, 0, 0};
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

    % Row 8: Advanced toggle button
    app.btnAdvanced = uibutton(paramGrid, 'Text', [char(9654), ' Advanced'], ...
        'ButtonPushedFcn', @onToggleAdvanced);
    app.btnAdvanced.Layout.Column = [1 2];

    % Row 9: mu (advanced, hidden by default)
    app.lblAdvMu = uilabel(paramGrid, 'Text', 'Penalty mu', 'Visible', 'off');
    app.edtMu = uieditfield(paramGrid, 'numeric', 'Value', 1e-3, ...
        'ValueDisplayFormat', '%.1e', 'Visible', 'off', ...
        'Tooltip', 'ADMM augmented Lagrangian penalty weight. (mu)');

    % Row 10: alpha (advanced, hidden by default)
    app.lblAdvAlpha = uilabel(paramGrid, 'Text', 'Regularize alpha', 'Visible', 'off');
    app.edtAlpha = uieditfield(paramGrid, 'numeric', 'Value', 0, ...
        'Visible', 'off', ...
        'Tooltip', 'Regularization parameter in global subproblem. (alpha)');

    % Row 11: ADMM_maxIter (advanced, hidden by default)
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

    %% ====== RIGHT: Color Range controls ======
    climGrid = uigridlayout(rightPanel, [1, 6]);
    climGrid.ColumnWidth = {80, 50, 60, 80, 60, '1x'};
    climGrid.Padding = [5 0 5 0];

    uilabel(climGrid, 'Text', 'Color Range:');
    app.cbAutoClim = uicheckbox(climGrid, 'Text', 'Auto', 'Value', true, ...
        'ValueChangedFcn', @onAutoClimChanged);
    uilabel(climGrid, 'Text', 'Min:', 'HorizontalAlignment', 'right');
    app.efClimMin = uieditfield(climGrid, 'numeric', 'Value', 0, ...
        'Enable', 'off', 'ValueChangedFcn', @(~,~) updateDisplay());
    uilabel(climGrid, 'Text', 'Max:', 'HorizontalAlignment', 'right');
    app.efClimMax = uieditfield(climGrid, 'numeric', 'Value', 1, ...
        'Enable', 'off', 'ValueChangedFcn', @(~,~) updateDisplay());

    %% ====== RIGHT: Progress + Log ======
    progGrid = uigridlayout(rightPanel, [2, 1]);
    progGrid.RowHeight = {45, '1x'};
    progGrid.Padding = [5 5 5 5];
    progGrid.RowSpacing = 5;

    app.gauge = uigauge(progGrid, 'linear', 'Limits', [0 100], 'Value', 0);
    app.gauge.ScaleColors = {[0.2 0.6 1]};
    app.gauge.ScaleColorLimits = [0 100];

    app.txtLog = uitextarea(progGrid, 'Value', {'Ready.'}, 'Editable', 'off');

    %% ======================================================
    %  Callback functions
    % =======================================================

    function onClose(~, ~)
        delete(app.fig);
    end

    function onLoadImages(~, ~)
        folder = uigetdir(pwd, 'Select images folder');
        if folder == 0, return; end
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
            imshow(app.Img{idx}', [], 'Parent', app.axMain);
            app.axMain.Title.String = sprintf('Image %d: %s', idx, app.file_name{1,idx});
        end
    end

    %% ====== Region callbacks (unified ROI/Mask) ======

    function onDrawRect(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        showImage(1);
        app.axMain.Title.String = 'Draw rectangle ROI, then double-click to confirm';
        roi = drawrectangle(app.axMain, 'Label', 'ROI');
        if isempty(roi.Position), return; end
        app.regionType = 'rect';
        app.regionRect = roi.Position;
        app.regionVertices = [];
        app.holeVertices = {};
        computeMaskFromShape();
        updateRegionInfo();
    end

    function onDrawPolygon(~, ~)
        if isempty(app.Img)
            uialert(app.fig, 'Load images first.', 'Error'); return;
        end
        showImage(1);
        app.axMain.Title.String = 'Draw polygon ROI, then double-click to close';
        roi = drawpolygon(app.axMain, 'Label', 'ROI');
        if isempty(roi.Position), return; end
        app.regionType = 'polygon';
        app.regionVertices = roi.Position;
        app.regionRect = [];
        app.holeVertices = {};
        computeMaskFromShape();
        updateRegionInfo();
    end

    function onAddHole(~, ~)
        if ~strcmp(app.regionType, 'polygon')
            uialert(app.fig, 'Draw a polygon first.', 'Add Hole'); return;
        end
        showMaskOverlay();
        app.axMain.Title.String = 'Draw hole polygon, then double-click to close';
        roi = drawpolygon(app.axMain, 'Label', 'Hole', 'Color', 'r');
        if isempty(roi.Position), return; end
        app.holeVertices{end+1} = roi.Position;
        computeMaskFromShape();
        updateRegionInfo();
    end

    function onImportMasks(~, ~)
        folder = uigetdir(pwd, 'Select mask files folder');
        if folder == 0, return; end
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
        app.ImgMask = cell(1, nFiles);
        for k = 1:nFiles
            fpath = fullfile(allFiles(k).folder, allFiles(k).name);
            app.mask_file_name{1,k} = allFiles(k).name;
            app.mask_file_name{2,k} = allFiles(k).folder;
            img = imread(fpath);
            [~, ~, nc] = size(img);
            if nc == 3, img = rgb2gray(img); end
            app.ImgMask{k} = logical(img)';
        end

        app.regionType = 'imported';
        app.regionRect = [];
        app.regionVertices = [];
        app.holeVertices = {};
        computROIFromImportedMask();
        updateRegionInfo();
        showMaskOverlay();
        logMsg(sprintf('Imported %d masks from %s', nFiles, folder));
    end

    function onClearRegion(~, ~)
        app.regionType = 'none';
        app.regionRect = [];
        app.regionVertices = [];
        app.holeVertices = {};
        app.ImgMask = {};
        app.roiGridx = [];
        app.roiGridy = [];
        app.btnAddHole.Enable = 'off';
        updateRegionInfo();
        if ~isempty(app.Img), showImage(1); end
    end

    function computeMaskFromShape()
        imgSize = size(app.Img{1}); % [originalCols, originalRows] (transposed)
        dispRows = imgSize(2);      % originalRows = display rows
        dispCols = imgSize(1);      % originalCols = display cols

        if strcmp(app.regionType, 'rect')
            pos = app.regionRect;
            displayMask = false(dispRows, dispCols);
            r1 = max(1, round(pos(2)));
            r2 = min(dispRows, round(pos(2)+pos(4)));
            c1 = max(1, round(pos(1)));
            c2 = min(dispCols, round(pos(1)+pos(3)));
            displayMask(r1:r2, c1:c2) = true;
        elseif strcmp(app.regionType, 'polygon')
            displayMask = poly2mask(app.regionVertices(:,1), ...
                app.regionVertices(:,2), dispRows, dispCols);
        else
            return;
        end

        % Subtract holes
        for k = 1:length(app.holeVertices)
            hv = app.holeVertices{k};
            holeMask = poly2mask(hv(:,1), hv(:,2), dispRows, dispCols);
            displayMask = displayMask & ~holeMask;
        end

        % Transpose to code space (matching Img{k} orientation)
        codeMask = logical(displayMask');

        % Replicate to all frames
        nFrames = length(app.Img);
        app.ImgMask = repmat({codeMask}, 1, nFrames);

        % Compute bounding rect for gridxyROIRange
        % gridx = dim 1 of code space (horizontal in display)
        % gridy = dim 2 of code space (vertical in display)
        [d1idx, d2idx] = find(codeMask);
        app.roiGridx = [min(d1idx), max(d1idx)];
        app.roiGridy = [min(d2idx), max(d2idx)];

        % Enable Add Hole only for polygon
        if strcmp(app.regionType, 'polygon')
            app.btnAddHole.Enable = 'on';
        else
            app.btnAddHole.Enable = 'off';
        end

        % Show mask overlay
        showMaskOverlay();
    end

    function computROIFromImportedMask()
        if isempty(app.ImgMask), return; end
        firstMask = app.ImgMask{1};
        [d1idx, d2idx] = find(firstMask);
        if isempty(d1idx), return; end
        app.roiGridx = [min(d1idx), max(d1idx)];
        app.roiGridy = [min(d2idx), max(d2idx)];
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
        app.axMain.Title.String = sprintf('Region: %s', app.regionType);
    end

    function updateRegionInfo()
        switch app.regionType
            case 'none'
                app.lblRegionStatus.Text = 'No region defined';
                app.lblROIInfo.Text = 'ROI: [ - ]';
            case 'rect'
                app.lblRegionStatus.Text = 'Rectangle drawn';
                app.lblROIInfo.Text = sprintf('ROI: x=[%d,%d]  y=[%d,%d]', ...
                    app.roiGridx(1), app.roiGridx(2), app.roiGridy(1), app.roiGridy(2));
            case 'polygon'
                nHoles = length(app.holeVertices);
                if nHoles > 0
                    app.lblRegionStatus.Text = sprintf('Polygon + %d hole(s)', nHoles);
                else
                    app.lblRegionStatus.Text = 'Polygon drawn';
                end
                app.lblROIInfo.Text = sprintf('ROI: x=[%d,%d]  y=[%d,%d]', ...
                    app.roiGridx(1), app.roiGridx(2), app.roiGridy(1), app.roiGridy(2));
            case 'imported'
                app.lblRegionStatus.Text = sprintf('%d masks imported', length(app.ImgMask));
                app.lblROIInfo.Text = sprintf('ROI: x=[%d,%d]  y=[%d,%d]', ...
                    app.roiGridx(1), app.roiGridx(2), app.roiGridy(1), app.roiGridy(2));
        end
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
            paramGrid.RowHeight(9:11) = {22, 22, 22};
            app.btnAdvanced.Text = [char(9660), ' Advanced'];
        else
            vis = 'off';
            paramGrid.RowHeight(9:11) = {0, 0, 0};
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
        if strcmp(app.regionType, 'none')
            uialert(app.fig, 'Define a region first (draw ROI or import masks).', 'Error'); return;
        end
        if strcmp(app.regionType, 'imported') && length(app.Img) ~= length(app.ImgMask)
            uialert(app.fig, sprintf('Image count (%d) != Mask count (%d).', ...
                length(app.Img), length(app.ImgMask)), 'Error'); return;
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
        app.gauge.Value = 0;
        logMsg('Starting DIC computation...');

        % Run pipeline
        try
            app.results = run_aldic(DICpara, app.file_name, app.Img, app.ImgMask, ...
                'ProgressFcn', @guiProgress, ...
                'StopFcn', @() app.stopRequested);
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
        app.gauge.Value = frac * 100;
        logMsg(msg);
        drawnow;
    end

    function logMsg(msg)
        currentLog = app.txtLog.Value;
        timestamp = datestr(now, 'HH:MM:SS');
        newLine = sprintf('[%s] %s', timestamp, msg);
        app.txtLog.Value = [currentLog; {newLine}];
        % Scroll to bottom
        app.txtLog.scroll('bottom');
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

    function updateDisplay()
        if isempty(app.results), return; end

        frameIdx = round(app.sldFrame.Value);
        fieldName = app.ddField.Value;

        if frameIdx < 1 || frameIdx > length(app.results.ResultStrain)
            return;
        end

        RS = app.results.ResultStrain{frameIdx};
        if isempty(RS), return; end

        % Select data field
        switch fieldName
            case 'Disp U',          data = RS.dispu;
            case 'Disp V',          data = RS.dispv;
            case 'exx',             data = RS.strain_exx;
            case 'exy',             data = RS.strain_exy;
            case 'eyy',             data = RS.strain_eyy;
            case 'vonMises',        data = RS.strain_vonMises;
            case 'maxshear',        data = RS.strain_maxshear;
            case 'principal max',   data = RS.strain_principal_max;
            case 'principal min',   data = RS.strain_principal_min;
            otherwise,              data = RS.dispu;
        end

        coordWorld = [RS.strainxCoord, RS.strainyCoord];
        elemFEM = app.results.ResultFEMeshEachFrame{1}.elementsFEM;

        % Plot using patch (more reliable in uiaxes than show())
        cla(app.axMain);

        % Build patch data for quadrilateral elements
        faces = elemFEM(:, 1:4);
        vertices = coordWorld;

        patch(app.axMain, 'Faces', faces, 'Vertices', vertices, ...
            'FaceVertexCData', data(:), ...
            'FaceColor', 'interp', 'EdgeColor', 'none');

        view(app.axMain, 2);
        axis(app.axMain, 'tight');
        axis(app.axMain, 'equal');
        set(app.axMain, 'YDir', 'normal');
        colorbar(app.axMain);
        colormap(app.axMain, 'jet');

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

end
