function test_case6_profile()
%test_case6_profile  Profile the STAQ-DIC pipeline on case6 metamaterial data.
%
%   Loads high-resolution experimental images (5472x3648, ~20 MP) with
%   complex mask geometry from Bolei Deng's metamaterial compression test.
%   Runs the full pipeline with a timing-instrumented ProgressFcn and
%   prints a detailed per-section timing breakdown to guide optimization.
%
%   Usage:
%       test_case6_profile          % run from STAQ-DIC-GUI root
%
% =========================================================================

close all; clc;
fprintf('============================================================\n');
fprintf('  STAQ-DIC Case 6 Profiling: Metamaterial (Bolei Deng)\n');
fprintf('============================================================\n\n');

%% 0. Setup paths
addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
    './third_party','./third_party/rbfinterp');

% Compile MEX if needed
try
    mex -O -outdir ./third_party ./third_party/ba_interp2_spline.cpp;
catch
end

%% 1. Locate data
caseDir = fullfile(pwd, 'test_data', 'case6_metamaterial_Bolei_Deng');
imgDir  = fullfile(caseDir, 'raw image');
maskDir = fullfile(caseDir, 'mask');
assert(exist(imgDir, 'dir') > 0, 'Image directory not found: %s', imgDir);
assert(exist(maskDir, 'dir') > 0, 'Mask directory not found: %s', maskDir);

% List images sorted by name (BMP files)
imgFiles  = dir(fullfile(imgDir,  '*.bmp'));
maskFiles = dir(fullfile(maskDir, '*.bmp'));
assert(~isempty(imgFiles),  'No BMP images found in %s', imgDir);
assert(~isempty(maskFiles), 'No BMP masks found in %s', maskDir);

% Match images and masks by filename
[~, imgNames]  = cellfun(@fileparts, {imgFiles.name},  'UniformOutput', false);
[~, maskNames] = cellfun(@fileparts, {maskFiles.name}, 'UniformOutput', false);
[~, idxImg, idxMask] = intersect(imgNames, maskNames, 'stable');
nFrames = length(idxImg);
fprintf('Found %d matched image/mask pairs.\n', nFrames);
assert(nFrames >= 2, 'Need at least 2 frames (1 reference + 1 deformed).');

% Pair by intersection indices, then sort by stem name
imgFiles  = imgFiles(idxImg);
maskFiles = maskFiles(idxMask);
[~, sortIdx] = sort(imgNames(idxImg));
imgFiles  = imgFiles(sortIdx);
maskFiles = maskFiles(sortIdx);

%% 2. Load images and masks (with timing)
fprintf('\n--- Loading %d frames ---\n', nFrames);
t_load = tic;

Img     = cell(1, nFrames);
ImgMask = cell(1, nFrames);

for i = 1:nFrames
    raw = double(imread(fullfile(imgFiles(i).folder, imgFiles(i).name)));
    % If RGB, convert to grayscale
    if ndims(raw) == 3
        raw = rgb2gray(uint8(raw));
        raw = double(raw);
    end
    Img{i} = raw';  % Transpose: file [H x W] -> code [W x H]

    rawMask = imread(fullfile(maskFiles(i).folder, maskFiles(i).name));
    if ndims(rawMask) == 3
        rawMask = rgb2gray(rawMask);
    end
    ImgMask{i} = logical(rawMask)';
    fprintf('  [%d/%d] %s  (%d x %d)\n', i, nFrames, imgFiles(i).name, size(raw,2), size(raw,1));
end

t_load_sec = toc(t_load);
fprintf('Image loading: %.1f s\n', t_load_sec);

% Report image and mask stats
imgSize = size(Img{1});
nMaskPixels = sum(ImgMask{1}(:));
totalPixels = prod(imgSize);
fprintf('Image size (code coords): %d x %d = %.1f MP\n', imgSize(1), imgSize(2), totalPixels/1e6);
fprintf('Mask coverage: %d / %d pixels (%.1f%%)\n', nMaskPixels, totalPixels, 100*nMaskPixels/totalPixels);

% Build file_name cell (required by run_aldic)
file_name = cell(6, nFrames);
for i = 1:nFrames
    file_name{1, i} = imgFiles(i).name;
    file_name{2, i} = imgFiles(i).folder;
end

%% 3. Build DICpara
fprintf('\n--- Configuring DICpara ---\n');
DICpara = struct();
DICpara.winsize      = 24;
DICpara.winstepsize  = 32;
DICpara.winsizeMin   = 32;
DICpara.ImgSize      = imgSize;
DICpara.showPlots    = false;
DICpara.NewFFTSearch = 1;
DICpara.referenceMode = 'incremental';
DICpara.ADMM_maxIter = 3;
DICpara.MethodToComputeStrain = 2;
DICpara.StrainPlaneFitRad = 20;
DICpara.SizeOfFFTSearchRegion = 10;

% Set ROI to full image (mask handles the geometry)
DICpara.gridxyROIRange.gridx = [1, imgSize(1)];
DICpara.gridxyROIRange.gridy = [1, imgSize(2)];

DICpara = dicpara_default(DICpara);

% Print key params
fprintf('  winsize       = %d\n', DICpara.winsize);
fprintf('  winstepsize   = %d\n', DICpara.winstepsize);
fprintf('  winsizeMin    = %d\n', DICpara.winsizeMin);
fprintf('  ADMM_maxIter  = %d\n', DICpara.ADMM_maxIter);
fprintf('  referenceMode = %s\n', DICpara.referenceMode);

% Estimate grid size
approxNodesX = floor((imgSize(1) - DICpara.winsize) / DICpara.winstepsize);
approxNodesY = floor((imgSize(2) - DICpara.winsize) / DICpara.winstepsize);
fprintf('  Approx uniform grid: %d x %d = %d nodes (before quadtree)\n', ...
    approxNodesX, approxNodesY, approxNodesX * approxNodesY);

%% 4. Run pipeline with timing instrumentation
fprintf('\n============================================================\n');
fprintf('  Running pipeline (this may take a long time)\n');
fprintf('============================================================\n\n');

% Timing state: shared workspace via nested function
timingLabels     = {};
timingTimestamps = [];
tGlobal = tic;

    function profiled_progress(frac, msg)
        elapsed = toc(tGlobal);
        timingLabels{end+1}     = msg;       %#ok<SETNU>
        timingTimestamps(end+1)  = elapsed;   %#ok<SETNU>
        fprintf('[%7.1fs | %3.0f%%] %s\n', elapsed, frac*100, msg);
    end

results = run_aldic(DICpara, file_name, Img, ImgMask, ...
    'ProgressFcn', @profiled_progress, 'ComputeStrain', true);

tTotal = toc(tGlobal);

%% 5. Print timing breakdown
fprintf('\n============================================================\n');
fprintf('  TIMING BREAKDOWN\n');
fprintf('============================================================\n\n');

% Compute per-section deltas from consecutive timestamps
nEvents = length(timingLabels);
fprintf('%-50s  %10s  %10s\n', 'Event', 'Wall (s)', 'Delta (s)');
fprintf('%s\n', repmat('-', 1, 74));
for i = 1:nEvents
    if i == 1
        delta = timingTimestamps(i);
    else
        delta = timingTimestamps(i) - timingTimestamps(i-1);
    end
    fprintf('%-50s  %10.1f  %10.1f\n', timingLabels{i}, timingTimestamps(i), delta);
end
fprintf('%s\n', repmat('-', 1, 74));
fprintf('%-50s  %10.1f\n', 'TOTAL', tTotal);

% Per-frame section summary (parse "Frame N: SX done" messages)
fprintf('\n--- Per-frame section times ---\n');
frameNums = [];
for i = 1:nEvents
    tok = regexp(timingLabels{i}, 'Frame (\d+): S(\d+) done', 'tokens');
    if ~isempty(tok)
        fnum = str2double(tok{1}{1});
        snum = str2double(tok{1}{2});
        if ~ismember(fnum, frameNums)
            frameNums(end+1) = fnum; %#ok<AGROW>
        end
    end
end

if ~isempty(frameNums)
    fprintf('\n%-8s', 'Frame');
    sectionNames = {'S3 FFT+mesh', 'S4 IC-GN', 'S5 Subpb2', 'S6 ADMM', 'Frame total'};
    for s = 1:length(sectionNames)
        fprintf('%14s', sectionNames{s});
    end
    fprintf('\n%s\n', repmat('-', 1, 8 + 14*length(sectionNames)));

    for fi = 1:length(frameNums)
        fnum = frameNums(fi);

        % Find frame start timestamp
        frameStartIdx = find(contains(timingLabels, sprintf('Processing frame %d/', fnum)), 1);
        if isempty(frameStartIdx), continue; end
        tFrameStart = timingTimestamps(frameStartIdx);

        % Find section-done timestamps for this frame (search AFTER frame start)
        sectionTimes = nan(1, 4);
        sectionEndTs = nan(1, 4);
        for sn = [3 4 5 6]
            pat = sprintf('Frame %d: S%d done', fnum, sn);
            for jj = frameStartIdx:nEvents
                if contains(timingLabels{jj}, pat)
                    sectionEndTs(sn - 2) = timingTimestamps(jj);
                    break;
                end
            end
        end

        % Compute deltas
        prevT = tFrameStart;
        for sn = 1:4
            if ~isnan(sectionEndTs(sn))
                sectionTimes(sn) = sectionEndTs(sn) - prevT;
                prevT = sectionEndTs(sn);
            end
        end
        frameTotal = prevT - tFrameStart;

        fprintf('%-8d', fnum);
        for sn = 1:4
            if isnan(sectionTimes(sn))
                fprintf('%14s', '-');
            else
                fprintf('%12.1f s', sectionTimes(sn));
            end
        end
        fprintf('%12.1f s', frameTotal);
        fprintf('\n');
    end
end

% Solver times from results struct
fprintf('\n--- Solver times from results struct ---\n');
fprintf('ALSub1Time (IC-GN per ADMM step):\n');
for i = 1:length(results.ALSub1Time)
    if results.ALSub1Time(i) > 0
        fprintf('  Step %d: %.1f s\n', i, results.ALSub1Time(i));
    end
end
fprintf('ALSub2Time (Subpb2 per ADMM step):\n');
for i = 1:length(results.ALSub2Time)
    if results.ALSub2Time(i) > 0
        fprintf('  Step %d: %.1f s\n', i, results.ALSub2Time(i));
    end
end
fprintf('Total ADMM steps: %d\n', results.ALSolveStep);

% Mesh stats
fprintf('\n--- Mesh statistics ---\n');
nNodesFinal = size(results.DICmesh.coordinatesFEM, 1);
nElemsFinal = size(results.DICmesh.elementsFEM, 1);
fprintf('Final mesh: %d nodes, %d elements\n', nNodesFinal, nElemsFinal);
fprintf('DOFs: %d (displacement) + %d (deformation gradient)\n', 2*nNodesFinal, 4*nNodesFinal);

% Memory estimate
memU = 2 * nNodesFinal * 8;  % double
memF = 4 * nNodesFinal * 8;
fprintf('Per-frame memory: U=%.1f MB, F=%.1f MB\n', memU/1e6, memF/1e6);

fprintf('\n============================================================\n');
fprintf('  PROFILING COMPLETE (total: %.1f s = %.1f min)\n', tTotal, tTotal/60);
fprintf('============================================================\n');

%% 6. Strain accuracy comparison: new FEM vs old plane-fit
fprintf('\n============================================================\n');
fprintf('  STRAIN ACCURACY: FEM vs Plane-Fit Comparison\n');
fprintf('============================================================\n\n');

% Reconstruct DICmesh/DICpara for S8 (same as run_aldic does)
coordinatesFEM = results.ResultFEMeshEachFrame{1}.coordinatesFEM;
elementsFEM    = results.ResultFEMeshEachFrame{1}.elementsFEM;
DICmeshS8 = results.DICmesh;
DICmeshS8.coordinatesFEM = coordinatesFEM;
DICmeshS8.elementsFEM    = elementsFEM;

DICparaS8 = results.DICpara;
DICparaS8.ImgRefMask = double(ImgMask{1});
DICparaS8.MethodToComputeStrain = 2;
DICparaS8.skipExtraSmoothing = 1;  % no extra smoothing, compare raw strain

% Mock Dg struct (compute_strain case 2 only uses Dg.imgSize for bounds)
Dg_mock.imgSize = DICparaS8.ImgSize;
Dg_mock.ImgRefMask = DICparaS8.ImgRefMask;

% Use Rad = 2 * winstepsize to ensure plane-fit finds enough neighbors.
% The default StrainPlaneFitRad=20 fails when winstepsize>=32 because
% rangesearch(20px) on a 32px grid finds no neighbors.
Rad = 2 * max(DICparaS8.winstepsize);
fprintf('Plane-fit Rad = %d px (2 * winstepsize=%d; default StrainPlaneFitRad=%d was too small)\n\n', ...
    Rad, max(DICparaS8.winstepsize), DICparaS8.StrainPlaneFitRad);

nCompareFrames = length(results.ResultStrain);
fprintf('Comparing %d frame(s)... (plane-fit will be slow)\n\n', nCompareFrames);

for fi = 1:nCompareFrames
    ULocal = results.ResultDisp{fi}.U_accum;

    % --- NEW method: FEM strain ---
    tNew = tic;
    FSubpb2_new = global_nodal_strain_fem(DICmeshS8, DICparaS8, ULocal);
    [FStrain_new, FStrainWorld_new] = apply_strain_type(FSubpb2_new, DICparaS8);
    tNewSec = toc(tNew);

    % --- OLD method: plane-fit via compute_strain ---
    fprintf('  Computing plane-fit strain for frame %d (this may take minutes)...\n', fi+1);
    tOld = tic;
    [FStrain_old, FStrainWorld_old] = compute_strain(ULocal, [], coordinatesFEM, ...
        DICmeshS8, DICparaS8, [], Dg_mock, Rad);
    tOldSec = toc(tOld);

    % --- Compare ---
    % Extract components (world coords)
    exx_new = FStrainWorld_new(1:4:end);  exx_old = FStrainWorld_old(1:4:end);
    exy_new = 0.5*(FStrainWorld_new(2:4:end) + FStrainWorld_new(3:4:end));
    exy_old = 0.5*(FStrainWorld_old(2:4:end) + FStrainWorld_old(3:4:end));
    eyy_new = FStrainWorld_new(4:4:end);  eyy_old = FStrainWorld_old(4:4:end);

    % Only compare nodes where both methods produced valid results
    validIdx = ~isnan(exx_old) & ~isnan(exx_new);
    nValid = sum(validIdx);
    nTotal = length(exx_old);
    nNanOld = sum(isnan(exx_old));
    nNanNew = sum(isnan(exx_new));

    diff_exx = exx_new(validIdx) - exx_old(validIdx);
    diff_exy = exy_new(validIdx) - exy_old(validIdx);
    diff_eyy = eyy_new(validIdx) - eyy_old(validIdx);

    fprintf('\n  Frame %d: %d nodes, %d valid, %d NaN_old, %d NaN_new\n', ...
        fi+1, nTotal, nValid, nNanOld, nNanNew);
    fprintf('  Timing: FEM=%.2f s, plane-fit=%.1f s (%.0fx speedup)\n\n', ...
        tNewSec, tOldSec, tOldSec/max(tNewSec, 0.001));

    if nValid == 0
        fprintf('  WARNING: no valid overlapping nodes. Skipping comparison.\n\n');
        continue;
    end

    fprintf('  Component    RMSE(new-old)   MAE(new-old)   max|diff|    mean(old)     std(old)\n');
    fprintf('  ----------   ------------   -----------   ---------   ----------   ----------\n');
    for ci = 1:3
        switch ci
            case 1, d = diff_exx; label = 'exx'; ref = exx_old(validIdx);
            case 2, d = diff_exy; label = 'exy'; ref = exy_old(validIdx);
            case 3, d = diff_eyy; label = 'eyy'; ref = eyy_old(validIdx);
        end
        fprintf('  %-12s %12.6f   %11.6f   %9.6f   %10.6f   %10.6f\n', ...
            label, rms(d), mean(abs(d)), max(abs(d)), mean(ref), std(ref));
    end

    % Relative error (normalized by strain range)
    range_exx = max(exx_old(validIdx)) - min(exx_old(validIdx));
    range_eyy = max(eyy_old(validIdx)) - min(eyy_old(validIdx));
    fprintf('  Relative RMSE: exx=%.2f%%, eyy=%.2f%% (of strain range)\n\n', ...
        100*rms(diff_exx)/max(range_exx, eps), 100*rms(diff_eyy)/max(range_eyy, eps));
end

fprintf('============================================================\n');
fprintf('  COMPARISON COMPLETE\n');
fprintf('============================================================\n');

end
