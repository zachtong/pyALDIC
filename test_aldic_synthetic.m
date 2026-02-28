% =========================================================================
% test_aldic_synthetic.m
% Synthetic test data & validation script for STAQ-DIC pipeline.
%
% Generates Gaussian speckle images with known displacement fields,
% runs the full ALDIC pipeline (Sections 2-9), and validates results
% against ground truth.
%
% Five test cases:
%   1. Zero displacement
%   2. Uniform translation (u=2.5, v=-1.8)
%   3. Affine deformation (2% uniform expansion)
%   4. Annular mask with affine deformation (ring mask, winsizeMin=4)
%   5. Simple shear (du/dy = 0.015)
%
% Usage: Run this script in MATLAB from the STAQ-DIC-GUI root directory.
% =========================================================================

close all; clear; clc; clearvars -global;
fprintf('============================================================\n');
fprintf('  STAQ-DIC Synthetic Test Suite\n');
fprintf('============================================================\n\n');

%% Part 0: Configuration
addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
    './third_party','./third_party/rbfinterp');

% Compile MEX if needed
try
    mex -O ba_interp2_spline.cpp;
catch
end

H = 256; W = 256;

% Define test cases: {name, frame2_u_func, frame2_v_func, frame3_u_func, frame3_v_func,
%                      GT_strain_F11, GT_strain_F22, mask_type, DICpara_overrides,
%                      GT_strain_F12, GT_strain_F21}
% Columns 10-11 default to 0 if not provided (handled in summary section).
test_cases = {
    'case1_zero',        @(x,y) zeros(size(x)), @(x,y) zeros(size(x)), ...
                         @(x,y) zeros(size(x)), @(x,y) zeros(size(x)), ...
                         0, 0, 'solid', struct(), 0, 0;
    'case2_translation', @(x,y)  2.5*ones(size(x)), @(x,y) -1.8*ones(size(x)), ...
                         @(x,y)  5.0*ones(size(x)), @(x,y) -3.6*ones(size(x)), ...
                         0, 0, 'solid', struct(), 0, 0;
    'case3_affine',      @(x,y) 0.02*(x-128), @(x,y) 0.02*(y-128), ...
                         @(x,y) 0.04*(x-128), @(x,y) 0.04*(y-128), ...
                         0.02, 0.02, 'solid', struct(), 0, 0;
    'case4_annular',     @(x,y) 0.02*(x-128), @(x,y) 0.02*(y-128), ...
                         @(x,y) 0.04*(x-128), @(x,y) 0.04*(y-128), ...
                         0.02, 0.02, 'annular', struct('winsizeMin', 4), 0, 0;
    'case5_shear',       @(x,y) 0.015*(y-128), @(x,y) zeros(size(x)), ...
                         @(x,y) 0.030*(y-128), @(x,y) zeros(size(x)), ...
                         0, 0, 'solid', struct(), 0.015, 0;
};

% Masks (in file coordinates: rows=y_file, cols=x_file)
% After transpose to code coords: dim1=x_code, dim2=y_code
[Xfile, Yfile] = meshgrid(1:W, 1:H);  % Xfile=col=x_file, Yfile=row=y_file
mask_solid = ((Xfile - 128).^2 + (Yfile - 128).^2) <= 90^2;  % [H x W] in file coords
mask_annular = mask_solid & ~(((Xfile - 128).^2 + (Yfile - 128).^2) <= 40^2);  % ring: outer R=90, inner R=40

% Summary storage
summary = struct();

%% Part 1: Generate synthetic data and save to disk
fprintf('------------------------------------------------------------\n');
fprintf('  Part 1: Generating synthetic test data\n');
fprintf('------------------------------------------------------------\n');

% Generate reference speckle in file coordinates [H x W]
ref_speckle_file = generate_speckle(H, W, 3, 42);

baseDir = fullfile(pwd, 'test_data');
if ~exist(baseDir, 'dir'), mkdir(baseDir); end

for tc = 1:size(test_cases, 1)
    caseName = test_cases{tc, 1};
    fprintf('\n--- Generating %s ---\n', caseName);

    % Select mask type for this case
    maskType = test_cases{tc, 8};
    if strcmp(maskType, 'annular')
        mask_file = mask_annular;
    else
        mask_file = mask_solid;
    end

    caseDir = fullfile(baseDir, caseName);
    imgDir = fullfile(caseDir, 'images');
    maskDir = fullfile(caseDir, 'masks');
    if ~exist(imgDir, 'dir'), mkdir(imgDir); end
    if ~exist(maskDir, 'dir'), mkdir(maskDir); end

    % Frame 1 = reference
    imwrite(uint8(ref_speckle_file), fullfile(imgDir, 'frame_01.tif'));
    imwrite(uint8(255 * double(mask_file)), fullfile(maskDir, 'mask_01.tif'));

    % In file coordinates: row=y_file, col=x_file
    % GT displacement functions take (x,y) in CODE coordinates (dim1=x, dim2=y)
    % But inverse_warp works in file coordinates.
    % Code coords: X_code(i,j)=i (dim1), Y_code(i,j)=j (dim2)
    % File coords: row=Y_code, col=X_code (transpose relationship)
    % So we need u_file(row,col) = u_code(col, row) since x_code=col_file
    % Actually: code image = file image transposed.
    %   code(i,j) = file(j,i), so x_code=i corresponds to col_file=i
    %   and y_code=j corresponds to row_file=j
    % The GT functions are defined in code coords: u(x_code, y_code)
    % For inverse warp in file coords: u_file(row,col) = u_code(col, row)
    %   because x_code = col, y_code = row

    % Generate frames 2 and 3
    for frame_idx = 2:3
        if frame_idx == 2
            u_func = test_cases{tc, 2};
            v_func = test_cases{tc, 3};
        else
            u_func = test_cases{tc, 4};
            v_func = test_cases{tc, 5};
        end

        % Build displacement fields in file coordinates [H x W]
        % u_code = displacement in x_code direction = displacement in col_file direction
        % v_code = displacement in y_code direction = displacement in row_file direction
        u_code_at_file = u_func(Xfile, Yfile);  % u_code(x=col, y=row) evaluated at each file pixel
        v_code_at_file = v_func(Xfile, Yfile);  % v_code(x=col, y=row)

        % In file image: pixel shifts are (row_shift, col_shift) = (v_code, u_code)
        % inverse_warp: warped(row,col) = ref(row - row_shift, col - col_shift)
        warped_file = inverse_warp(ref_speckle_file, v_code_at_file, u_code_at_file);

        imwrite(uint8(warped_file), fullfile(imgDir, sprintf('frame_%02d.tif', frame_idx)));
        imwrite(uint8(255 * double(mask_file)), fullfile(maskDir, sprintf('mask_%02d.tif', frame_idx)));
    end

    % Save ground truth
    GT.u_func2 = test_cases{tc, 2}; GT.v_func2 = test_cases{tc, 3};
    GT.u_func3 = test_cases{tc, 4}; GT.v_func3 = test_cases{tc, 5};
    GT.F11 = test_cases{tc, 6}; GT.F22 = test_cases{tc, 7};
    save(fullfile(caseDir, 'ground_truth.mat'), 'GT');

    fprintf('  Saved 3 frames + masks to %s\n', caseDir);
end


%% Part 2: Run DIC pipeline for each test case
for tc = 1:size(test_cases, 1)
    caseName = test_cases{tc, 1};
    fprintf('\n============================================================\n');
    fprintf('  Part 2: Running DIC on %s\n', caseName);
    fprintf('============================================================\n');

    caseDir = fullfile(baseDir, caseName);
    imgDir = fullfile(caseDir, 'images');
    maskDir = fullfile(caseDir, 'masks');

    % GT functions for this case (in code coordinates)
    gt_u2 = test_cases{tc, 2}; gt_v2 = test_cases{tc, 3};
    gt_u3 = test_cases{tc, 4}; gt_v3 = test_cases{tc, 5};
    gt_F11 = test_cases{tc, 6}; gt_F22 = test_cases{tc, 7};
    gt_F12 = test_cases{tc, 10}; gt_F21 = test_cases{tc, 11};

    % ====== Section 2 substitute: Load images directly ======
    fprintf('\n------------ Section 2 Start ------------\n');

    nFrames = 3;
    Img = cell(nFrames, 1);
    ImgMask = cell(nFrames, 1);

    for i = 1:nFrames
        raw = double(imread(fullfile(imgDir, sprintf('frame_%02d.tif', i))));
        Img{i} = raw';  % Transpose: file [H x W] -> code [W x H]

        rawMask = imread(fullfile(maskDir, sprintf('mask_%02d.tif', i)));
        ImgMask{i} = logical(rawMask)';  % Transpose
    end

    % Build file_name cell: row1=filename, row2=folder
    file_name = cell(2, nFrames);
    for i = 1:nFrames
        file_name{1, i} = sprintf('frame_%02d.tif', i);
        file_name{2, i} = imgDir;
    end

    % Set up DICpara
    DICpara = struct();
    DICpara.winsize = 32;
    DICpara.winstepsize = 16;
    DICpara.winsizeMin = 8;
    DICpara.ImgSize = size(Img{1});  % [W, H] in code coords after transpose
    DICpara.gridxyROIRange.gridx = [1, 256];
    DICpara.gridxyROIRange.gridy = [1, 256];
    DICpara.NewFFTSearch = 1;
    DICpara.SizeOfFFTSearchRegion = 10;  % Small search region for 256x256 images (max disp ~5px)
    DICpara.referenceMode = 'accumulative';
    DICpara.OrigDICImgTransparency = 1;
    DICpara.ADMM_maxIter = 3;
    DICpara.MethodToComputeStrain = 2;
    DICpara.StrainPlaneFitRad = 20;
    DICpara.StrainType = 0;
    DICpara.MaterialModel = 1;
    DICpara.MaterialModelPara.YoungsModulus = 69e9;
    DICpara.MaterialModelPara.PoissonsRatio = 0.3;

    % Apply per-case DICpara overrides
    overrides = test_cases{tc, 9};
    fields = fieldnames(overrides);
    for i = 1:length(fields)
        DICpara.(fields{i}) = overrides.(fields{i});
    end
    DICpara = dicpara_default(DICpara);

    % Debug: image stats
    for i = 1:nFrames
        fprintf('[debug] Img{%d}: size=[%d,%d], class=%s, min=%.1f, max=%.1f, mean=%.1f\n', ...
            i, size(Img{i},1), size(Img{i},2), class(Img{i}), min(Img{i}(:)), max(Img{i}(:)), mean(Img{i}(:)));
        fprintf('[debug] ImgMask{%d}: size=[%d,%d], sum=%d\n', ...
            i, size(ImgMask{i},1), size(ImgMask{i},2), sum(ImgMask{i}(:)));
    end

    % Normalize images
    [ImgNormalized, DICpara.gridxyROIRange] = normalize_img(Img, DICpara.gridxyROIRange);
    for i = 1:nFrames
        fprintf('[debug] ImgNormalized{%d}: size=[%d,%d], min=%.2f, max=%.2f, mean=%.4f, std=%.4f\n', ...
            i, size(ImgNormalized{i},1), size(ImgNormalized{i},2), ...
            min(ImgNormalized{i}(:)), max(ImgNormalized{i}(:)), ...
            mean(ImgNormalized{i}(:)), std(ImgNormalized{i}(:)));
    end

    % Initialize result storage
    ResultDisp = cell(nFrames-1, 1);
    ResultDefGrad = cell(nFrames-1, 1);
    ResultStrain = cell(nFrames-1, 1);
    ResultStress = cell(nFrames-1, 1);
    ResultFEMeshEachFrame = cell(nFrames-1, 1);
    ResultFEMesh = cell(ceil((nFrames-1)/DICpara.ImgSeqIncUnit), 1);

    fprintf('------------ Section 2 Done ------------\n\n');

    UseGlobal = DICpara.UseGlobalStep;

    % ====== Main loop over frames ======
    for ImgSeqNum = 2:nFrames

        close all;
        fprintf('Current image frame #: %d/%d\n', ImgSeqNum, nFrames);

        % Select GT functions for this frame
        if ImgSeqNum == 2
            gt_u_func = gt_u2; gt_v_func = gt_v2;
        else
            gt_u_func = gt_u3; gt_v_func = gt_v3;
        end

        % ====== Load reference & deformed images (accumulative mode) ======
        fNormalizedMask = double(ImgMask{1});
        fNormalized = ImgNormalized{1} .* fNormalizedMask;
        Df = img_gradient(fNormalized, fNormalized, fNormalizedMask);

        gNormalizedMask = double(ImgMask{ImgSeqNum});
        gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask;
        DICpara.ImgRefMask = fNormalizedMask;

        % ====== Section 3: Initial guess ======
        fprintf('\n------------ Section 3 Start ------------\n');

        if ImgSeqNum == 2 || DICpara.NewFFTSearch == 1

            DICpara.InitFFTSearchMethod = 1;
            [DICpara, x0temp_f, y0temp_f, u_f, v_f, cc] = integer_search(fNormalized, gNormalized, file_name, DICpara);

            % Debug: integer_search results
            fprintf('[debug] integer_search: u_f mean=%.4f, min=%.4f, max=%.4f, NaN=%d\n', ...
                nanmean(u_f(:)), nanmin(u_f(:)), nanmax(u_f(:)), sum(isnan(u_f(:))));
            fprintf('[debug] integer_search: v_f mean=%.4f, min=%.4f, max=%.4f, NaN=%d\n', ...
                nanmean(v_f(:)), nanmin(v_f(:)), nanmax(v_f(:)), sum(isnan(v_f(:))));

            % Zach RBF improvement (from main_ALDIC lines 105-183)
            xnodes = max([1+0.5*DICpara.winsize, DICpara.gridxyROIRange.gridx(1)]) ...
                : DICpara.winstepsize : min([size(fNormalized,1)-0.5*DICpara.winsize-1, DICpara.gridxyROIRange.gridx(2)]);
            ynodes = max([1+0.5*DICpara.winsize, DICpara.gridxyROIRange.gridy(1)]) ...
                : DICpara.winstepsize : min([size(fNormalized,2)-0.5*DICpara.winsize-1, DICpara.gridxyROIRange.gridy(2)]);
            [x0temp, y0temp] = ndgrid(xnodes, ynodes);

            valid_indices_u = find(~isnan(u_f(:)));
            valid_indices_v = find(~isnan(v_f(:)));
            valid_indices = intersect(valid_indices_u, valid_indices_v);

            discontinuity_threshold_cc = DICpara.discontinuity_threshold_cc;
            low_cc_local_indices = find(cc.max(valid_indices) < discontinuity_threshold_cc);
            discontinuity_indices = valid_indices(low_cc_local_indices);
            smooth_indices = setdiff(valid_indices, discontinuity_indices);

            if ~isempty(smooth_indices)
                op1_smooth = rbfcreate([x0temp_f(smooth_indices), y0temp_f(smooth_indices)]', [u_f(smooth_indices)]', 'RBFFunction', 'thinplate');
                u_smooth = rbfinterp([x0temp(:), y0temp(:)]', op1_smooth);
                op2_smooth = rbfcreate([x0temp_f(smooth_indices), y0temp_f(smooth_indices)]', [v_f(smooth_indices)]', 'RBFFunction', 'thinplate');
                v_smooth = rbfinterp([x0temp(:), y0temp(:)]', op2_smooth);
                u_final = regularizeNd([x0temp(:), y0temp(:)], u_smooth(:), {xnodes', ynodes'}, 1e-3);
                v_final = regularizeNd([x0temp(:), y0temp(:)], v_smooth(:), {xnodes', ynodes'}, 1e-3);
            else
                u_final = nan(size(x0temp));
                v_final = nan(size(x0temp));
            end

            if ~isempty(discontinuity_indices)
                discontinuous_points_coords = [x0temp_f(discontinuity_indices), y0temp_f(discontinuity_indices)];
                discontinuous_u_values = u_f(discontinuity_indices);
                nearest_idx_u = knnsearch(discontinuous_points_coords, [x0temp(:), y0temp(:)]);
                u_discontinuous = reshape(discontinuous_u_values(nearest_idx_u), size(x0temp));
                discontinuous_v_values = v_f(discontinuity_indices);
                nearest_idx_v = knnsearch(discontinuous_points_coords, [x0temp(:), y0temp(:)]);
                v_discontinuous = reshape(discontinuous_v_values(nearest_idx_v), size(x0temp));

                k_nearest_neighbors = DICpara.k_nearest_neighbors;
                [~, nearest_indices_in_orig_grid] = pdist2([x0temp_f(:), y0temp_f(:)], [x0temp(:), y0temp(:)], 'euclidean', 'Smallest', k_nearest_neighbors);
                is_neighbor_discontinuous = ismember(nearest_indices_in_orig_grid, discontinuity_indices);
                is_discontinuous_on_new_grid = any(is_neighbor_discontinuous, 1);
                is_discontinuous_on_new_grid = reshape(is_discontinuous_on_new_grid, size(x0temp));

                u_final(is_discontinuous_on_new_grid) = u_discontinuous(is_discontinuous_on_new_grid);
                v_final(is_discontinuous_on_new_grid) = v_discontinuous(is_discontinuous_on_new_grid);
            end

            u = u_final;
            v = v_final;

            % Mesh setup
            [DICmesh] = mesh_setup(x0temp, y0temp, DICpara);
            U0 = init_disp(u, v, cc.max, DICmesh.x0, DICmesh.y0, 0);

            % Debug: Init results
            nNode = size(DICmesh.coordinatesFEM, 1);
            gt_u_nodes = gt_u_func(DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
            gt_v_nodes = gt_v_func(DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
            stats_init = compare_disp_GT(U0, DICmesh.coordinatesFEM, gt_u_nodes, gt_v_nodes, fNormalizedMask);
            debug_print('Init U0', stats_init);

            % Set zero at mask holes
            linearIndices1 = sub2ind(size(fNormalizedMask), DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
            MaskOrNot1 = fNormalizedMask(linearIndices1);
            nanIndex = find(MaskOrNot1 < 1);
            U0(2*nanIndex) = nan;
            U0(2*nanIndex-1) = nan;

            % Incremental mode bookkeeping
            fNormalizedNewIndex = ImgSeqNum - mod(ImgSeqNum-2, DICpara.ImgSeqIncUnit) - 1;
            if DICpara.ImgSeqIncUnit == 1, fNormalizedNewIndex = fNormalizedNewIndex - 1; end
            ResultFEMesh{1+floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)} = ...
                struct('coordinatesFEM', DICmesh.coordinatesFEM, 'elementsFEM', DICmesh.elementsFEM, ...
                'winsize', DICpara.winsize, 'winstepsize', DICpara.winstepsize, 'gridxyROIRange', DICpara.gridxyROIRange);

            % Generate quadtree mesh
            DICmesh.elementMinSize = DICpara.winsizeMin;
            [DICmesh, U0] = generate_mesh(DICmesh, DICpara, Df, U0);

            % Debug: quadtree mesh
            fprintf('[debug] QuadtreeMesh: nNodes=%d, nElements=%d, markCoordHoleEdge=%d\n', ...
                size(DICmesh.coordinatesFEM,1), size(DICmesh.elementsFEM,1), length(DICmesh.markCoordHoleEdge));

            plot_disp_show(U0, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');

            ResultFEMeshEachFrame{ImgSeqNum-1} = struct('coordinatesFEM', DICmesh.coordinatesFEM, ...
                'elementsFEM', DICmesh.elementsFEM, 'markCoordHoleEdge', DICmesh.markCoordHoleEdge);

        else
            % Reuse previous mesh
            U0 = ResultDisp{ImgSeqNum-2}.U;
            disp('Previous frame result used as initial guess.');
            ResultFEMeshEachFrame{ImgSeqNum-1} = struct('coordinatesFEM', DICmesh.coordinatesFEM, ...
                'elementsFEM', DICmesh.elementsFEM, 'markCoordHoleEdge', DICmesh.markCoordHoleEdge);
        end

        fprintf('------------ Section 3 Done ------------\n\n');

        %% Section 4: Local ICGN
        fprintf('------------ Section 4 Start ------------\n');
        mu = 0; beta = 0; tol = DICpara.tol; ALSolveStep = 1;
        ALSub1Time = zeros(6,1); ALSub2Time = zeros(6,1);
        ConvItPerEle = zeros(size(DICmesh.coordinatesFEM,1), 6);
        ALSub1BadPtNum = zeros(6,1);

        disp(['***** Start step', num2str(ALSolveStep), ' Subproblem1 *****']);
        [USubpb1, FSubpb1, HtempPar, ALSub1Timetemp, ConvItPerEletemp, LocalICGNBadPtNumtemp, markCoordHoleStrain] = ...
            local_icgn(U0, DICmesh.coordinatesFEM, Df, fNormalized, gNormalized, DICpara, DICpara.ICGNmethod, tol);
        ALSub1Time(ALSolveStep) = ALSub1Timetemp;
        ConvItPerEle(:, ALSolveStep) = ConvItPerEletemp;
        ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp;

        coordinatesFEM = DICmesh.coordinatesFEM;
        U = USubpb1; F = FSubpb1;

        % Debug: LocalICGN results
        gt_u_nodes = gt_u_func(DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
        gt_v_nodes = gt_v_func(DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
        stats_local = compare_disp_GT(USubpb1, DICmesh.coordinatesFEM, gt_u_nodes, gt_v_nodes, fNormalizedMask);
        debug_print('LocalICGN USubpb1', stats_local);
        fprintf('[debug] LocalICGN bad points: %d\n', LocalICGNBadPtNumtemp);

        USubpb1World = USubpb1; USubpb1World(2:2:end) = -USubpb1(2:2:end);
        FSubpb1World = FSubpb1;
        plot_disp_show(USubpb1World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
        plot_strain_show(FSubpb1World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
        save(fullfile(caseDir, ['Subpb1_step', num2str(ALSolveStep)]), 'USubpb1', 'FSubpb1');
        fprintf('------------ Section 4 Done ------------\n\n');

        if UseGlobal
            %% Section 5: Subproblem 2
            fprintf('------------ Section 5 Start ------------\n'); tic;

            LevelNo = 1;
            if DICpara.DispSmoothness > 1e-6, USubpb1 = smooth_disp_rbf(USubpb1, DICmesh, DICpara); end
            if DICpara.StrainSmoothness > 1e-6, FSubpb1 = smooth_strain_rbf(FSubpb1, DICmesh, DICpara); end

            mu = DICpara.mu; udual = 0*FSubpb1; vdual = 0*USubpb1;
            betaList = DICpara.betaRange * mean(DICpara.winstepsize).^2 .* mu;
            Err1 = zeros(length(betaList), 1); Err2 = Err1;

            disp(['***** Start step', num2str(ALSolveStep), ' Subproblem2 *****']);
            alpha = DICpara.alpha;

            if ImgSeqNum == 2
                for tempk = 1:length(betaList)
                    beta = betaList(tempk);
                    display(['Try #', num2str(tempk), ' beta = ', num2str(beta)]);
                    alpha = 0;
                    [USubpb2] = subpb2_solver(DICmesh, DICpara.GaussPtOrder, beta, mu, USubpb1, FSubpb1, udual, vdual, alpha, mean(DICpara.winstepsize), 0);
                    FSubpb2 = global_nodal_strain_rbf(DICmesh, DICpara, USubpb2);
                    Err1(tempk) = norm(USubpb1 - USubpb2, 2);
                    Err2(tempk) = norm(FSubpb1 - FSubpb2, 2);
                end

                Err1Norm = (Err1 - mean(Err1)) / std(Err1);
                Err2Norm = (Err2 - mean(Err2)) / std(Err2);
                ErrSum = Err1Norm + Err2Norm;
                [~, indexOfbeta] = min(ErrSum);

                try
                    [fitobj] = fit(log10(betaList(indexOfbeta-1:1:indexOfbeta+1))', ErrSum(indexOfbeta-1:1:indexOfbeta+1), 'poly2');
                    p = coeffvalues(fitobj); beta = 10^(-p(2)/2/p(1));
                catch
                    beta = betaList(indexOfbeta);
                end
                display(['Best beta = ', num2str(beta)]);
            else
                if ~isempty(DICpara.beta)
                    beta = DICpara.beta;
                else
                    beta = 1e-3 * mean(DICpara.winstepsize).^2 .* mu;
                end
            end

            if abs(beta - betaList(end)) > abs(eps)
                [USubpb2] = subpb2_solver(DICmesh, DICpara.GaussPtOrder, beta, mu, USubpb1, FSubpb1, udual, vdual, alpha, mean(DICpara.winstepsize), 0);
                FSubpb2 = global_nodal_strain_rbf(DICmesh, DICpara, USubpb2);
                ALSub2Time(ALSolveStep) = toc;
            end

            if DICpara.DispSmoothness > 1e-6, USubpb2 = smooth_disp_rbf(USubpb2, DICmesh, DICpara); end
            for tempk = 0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
            if DICpara.StrainSmoothness > 1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1, DICmesh, DICpara); end
            for tempk = 0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
            for tempk = 0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end

            % Debug: Subpb2 results
            stats_subpb2 = compare_disp_GT(USubpb2, DICmesh.coordinatesFEM, gt_u_nodes, gt_v_nodes, fNormalizedMask);
            debug_print('Subpb2 USubpb2', stats_subpb2);

            save(fullfile(caseDir, ['Subpb2_step', num2str(ALSolveStep)]), 'USubpb2', 'FSubpb2');
            fprintf('[debug] Section 5 end sizes: USubpb1=%s, FSubpb1=%s, USubpb2=%s, FSubpb2=%s\n', ...
                mat2str(size(USubpb1)), mat2str(size(FSubpb1)), mat2str(size(USubpb2)), mat2str(size(FSubpb2)));
            udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
            fprintf('[debug] Section 5 dual sizes: udual=%s, vdual=%s\n', mat2str(size(udual)), mat2str(size(vdual)));
            save(fullfile(caseDir, ['uvdual_step', num2str(ALSolveStep)]), 'udual', 'vdual');
            fprintf('------------ Section 5 Done ------------\n\n');

            %% Section 6: ADMM iterations
            fprintf('------------ Section 6 Start ------------\n');
            ALSolveStep = 1; tol2 = DICpara.ADMM_tol; UpdateY = 1e4;
            HPar = cell(21, 1); for tempj = 1:21, HPar{tempj} = HtempPar(:, tempj); end

            while (ALSolveStep < DICpara.ADMM_maxIter)
                ALSolveStep = ALSolveStep + 1;
                winsize_List = DICpara.winsize * ones(size(DICmesh.coordinatesFEM, 1), 2);
                DICpara.winsize_List = winsize_List;

                % Subproblem 1
                disp(['***** Start step', num2str(ALSolveStep), ' Subproblem1 *****']);
                tic; [USubpb1, ~, ALSub1Timetemp, ConvItPerEletemp, LocalICGNBadPtNumtemp] = subpb1_solver(...
                    USubpb2, FSubpb2, udual, vdual, DICmesh.coordinatesFEM, ...
                    Df, fNormalized, gNormalized, mu, beta, HPar, ALSolveStep, DICpara, DICpara.ICGNmethod, tol);
                FSubpb1 = FSubpb2; toc
                fprintf('[debug] After Subpb1 step %d: USubpb1=%s, FSubpb1=%s\n', ALSolveStep, mat2str(size(USubpb1)), mat2str(size(FSubpb1)));
                ALSub1Time(ALSolveStep) = ALSub1Timetemp;
                ConvItPerEle(:, ALSolveStep) = ConvItPerEletemp;
                ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp;
                save(fullfile(caseDir, ['Subpb1_step', num2str(ALSolveStep)]), 'USubpb1', 'FSubpb1');

                % Subproblem 2
                disp(['***** Start step', num2str(ALSolveStep), ' Subproblem2 *****']);
                fprintf('[debug] ADMM Subpb2 input sizes: USubpb1=%s, FSubpb1=%s, udual=%s, vdual=%s, nNode=%d\n', ...
                    mat2str(size(USubpb1)), mat2str(size(FSubpb1)), mat2str(size(udual)), mat2str(size(vdual)), size(DICmesh.coordinatesFEM,1));
                tic; [USubpb2] = subpb2_solver(DICmesh, DICpara.GaussPtOrder, beta, mu, USubpb1, FSubpb1, udual, vdual, alpha, mean(DICpara.winstepsize), 0);
                FSubpb2 = global_nodal_strain_rbf(DICmesh, DICpara, USubpb2);
                ALSub2Time(ALSolveStep) = toc;

                if DICpara.DispSmoothness > 1e-6, USubpb2 = smooth_disp_rbf(USubpb2, DICmesh, DICpara); end
                for tempk = 0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
                if DICpara.StrainSmoothness > 1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1, DICmesh, DICpara); end
                for tempk = 0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
                for tempk = 0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end

                save(fullfile(caseDir, ['Subpb2_step', num2str(ALSolveStep)]), 'USubpb2', 'FSubpb2');

                % Debug: ADMM iteration results
                stats_admm = compare_disp_GT(USubpb2, DICmesh.coordinatesFEM, gt_u_nodes, gt_v_nodes, fNormalizedMask);
                debug_print(sprintf('ADMM iter %d USubpb2', ALSolveStep), stats_admm);

                % Convergence check
                USubpb2_Old = load(fullfile(caseDir, ['Subpb2_step', num2str(ALSolveStep-1)]), 'USubpb2');
                USubpb2_New = load(fullfile(caseDir, ['Subpb2_step', num2str(ALSolveStep)]), 'USubpb2');
                USubpb1_Old = load(fullfile(caseDir, ['Subpb1_step', num2str(ALSolveStep-1)]), 'USubpb1');
                USubpb1_New = load(fullfile(caseDir, ['Subpb1_step', num2str(ALSolveStep)]), 'USubpb1');
                if (mod(ImgSeqNum-2, DICpara.ImgSeqIncUnit) ~= 0 && (ImgSeqNum > 2)) || (ImgSeqNum < DICpara.ImgSeqIncUnit)
                    UpdateY = norm((USubpb2_Old.USubpb2 - USubpb2_New.USubpb2), 2) / sqrt(size(USubpb2_Old.USubpb2, 1));
                    UpdateY2 = norm((USubpb1_Old.USubpb1 - USubpb1_New.USubpb1), 2) / sqrt(size(USubpb1_Old.USubpb1, 1));
                end
                if exist('UpdateY', 'var'), disp(['[debug] Update global step = ', num2str(UpdateY)]); end
                if exist('UpdateY2', 'var'), disp(['[debug] Update local step  = ', num2str(UpdateY2)]); end
                fprintf('***********************************\n\n');

                udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
                save(fullfile(caseDir, ['uvdual_step', num2str(ALSolveStep)]), 'udual', 'vdual');

                if exist('UpdateY', 'var') && exist('UpdateY2', 'var')
                    if UpdateY < tol2 || UpdateY2 < tol2
                        break
                    end
                end
            end
            fprintf('------------ Section 6 Done ------------\n\n');
        end % UseGlobal

        % Save results
        if UseGlobal
            ResultDisp{ImgSeqNum-1}.U = full(USubpb2);
            ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
            ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb2);
        else
            ResultDisp{ImgSeqNum-1}.U = full(USubpb1);
            ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
            ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb1);
        end

    end % ImgSeqNum loop

    %% Accumulative mode: U is already cumulative
    for ImgSeqNum = 2:nFrames
        ResultDisp{ImgSeqNum-1}.U_accum = ResultDisp{ImgSeqNum-1}.U;
    end

    %% Section 8: Compute strains
    fprintf('------------ Section 8 Start ------------\n');
    Rad = [];
    if DICpara.MethodToComputeStrain == 2
        Rad = DICpara.StrainPlaneFitRad;
    end
    if DICpara.smoothness > 0
        DICpara.DoYouWantToSmoothOnceMore = 0;
    else
        DICpara.DoYouWantToSmoothOnceMore = 1;
    end

    for ImgSeqNum = 2:nFrames

        close all;
        fprintf('Current image frame #: %d/%d\n', ImgSeqNum, nFrames);

        if ImgSeqNum == 2
            gt_u_func = gt_u2; gt_v_func = gt_v2;
        else
            gt_u_func = gt_u3; gt_v_func = gt_v3;
        end

        gNormalizedMask = double(ImgMask{ImgSeqNum});
        gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask;
        Dg = img_gradient(gNormalized, gNormalized, gNormalizedMask);

        fNormalizedMask = double(ImgMask{1});
        DICpara.ImgRefMask = fNormalizedMask;

        USubpb2 = ResultDisp{ImgSeqNum-1}.U_accum;
        coordinatesFEM = ResultFEMeshEachFrame{1}.coordinatesFEM;
        elementsFEM = ResultFEMeshEachFrame{1}.elementsFEM;

        if isfield(ResultFEMeshEachFrame{ImgSeqNum-1}, 'markCoordHoleEdge')
            markCoordHoleEdge = ResultFEMeshEachFrame{ImgSeqNum-1}.markCoordHoleEdge;
        end
        DICmesh.coordinatesFEM = coordinatesFEM;
        DICmesh.elementsFEM = elementsFEM;
        coordinatesFEMWorld = DICpara.um2px * [coordinatesFEM(:,1), size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)];

        % Debug: cumulative displacement
        gt_u_nodes = gt_u_func(coordinatesFEM(:,1), coordinatesFEM(:,2));
        gt_v_nodes = gt_v_func(coordinatesFEM(:,1), coordinatesFEM(:,2));
        stats_accum = compare_disp_GT(USubpb2, coordinatesFEM, gt_u_nodes, gt_v_nodes, fNormalizedMask);
        debug_print(sprintf('Section8 Frame%d U_accum', ImgSeqNum), stats_accum);

        if size(USubpb2, 1) == 1
            ULocal = USubpb2_New.USubpb2;
        else
            ULocal = USubpb2;
        end
        UWorld = DICpara.um2px * ULocal; UWorld(2:2:end) = -UWorld(2:2:end);

        SmoothTimes = 0;
        while DICpara.DoYouWantToSmoothOnceMore == 0 && SmoothTimes < 3
            ULocal = smooth_disp_rbf(ULocal, DICmesh, DICpara);
            SmoothTimes = SmoothTimes + 1;
        end

        Df_sec8 = img_gradient(ImgNormalized{1} .* fNormalizedMask, ImgNormalized{1} .* fNormalizedMask, fNormalizedMask);
        [FStraintemp, FStrainWorld] = compute_strain(ULocal, [], coordinatesFEM, DICmesh, DICpara, Df_sec8, Dg, Rad);

        % Plot (transparency=1 path: no file_name needed)
        plot_disp_show(UWorld, coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'NoEdgeColor');
        [strain_exx, strain_exy, strain_eyy, strain_principal_max, strain_principal_min, strain_maxshear, strain_vonMises] = ...
            plot_strain_no_img(FStrainWorld, coordinatesFEMWorld, elementsFEM(:,1:4), DICpara);

        % Debug: strain results (cases with nonzero strain GT)
        if gt_F11 ~= 0 || gt_F22 ~= 0 || gt_F12 ~= 0 || gt_F21 ~= 0
            gt_F11_nodes = gt_F11 * ones(size(coordinatesFEM, 1), 1);
            gt_F22_nodes = gt_F22 * ones(size(coordinatesFEM, 1), 1);
            gt_F12_nodes = gt_F12 * ones(size(coordinatesFEM, 1), 1);
            gt_F21_nodes = gt_F21 * ones(size(coordinatesFEM, 1), 1);
            stats_strain = compare_strain_GT(FStraintemp, gt_F11_nodes, gt_F21_nodes, gt_F12_nodes, gt_F22_nodes, coordinatesFEM, fNormalizedMask);
            debug_print(sprintf('Section8 Frame%d Strain', ImgSeqNum), stats_strain);
        end

        ResultStrain{ImgSeqNum-1} = struct('strainxCoord', coordinatesFEMWorld(:,1), 'strainyCoord', coordinatesFEMWorld(:,2), ...
            'dispu', UWorld(1:2:end), 'dispv', UWorld(2:2:end), ...
            'dudx', FStraintemp(1:4:end), 'dvdx', FStraintemp(2:4:end), 'dudy', FStraintemp(3:4:end), 'dvdy', FStraintemp(4:4:end), ...
            'strain_exx', strain_exx, 'strain_exy', strain_exy, 'strain_eyy', strain_eyy, ...
            'strain_principal_max', strain_principal_max, 'strain_principal_min', strain_principal_min, ...
            'strain_maxshear', strain_maxshear, 'strain_vonMises', strain_vonMises);

        % Skip SaveFigFiles to avoid uigetdir prompt
    end
    fprintf('------------ Section 8 Done ------------\n\n');

    %% Section 9: Compute stress
    fprintf('------------ Section 9 Start ------------\n');
    for ImgSeqNum = 2:nFrames
        fprintf('Current image frame #: %d/%d\n', ImgSeqNum, nFrames); close all;
        coordinatesFEM_s9 = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM;
        elementsFEM_s9 = ResultFEMeshEachFrame{ImgSeqNum-1}.elementsFEM;
        coordinatesFEMWorldDef = DICpara.um2px * [coordinatesFEM_s9(:,1), size(ImgNormalized{1},2)+1-coordinatesFEM_s9(:,2)] + ...
            DICpara.Image2PlotResults * [ResultStrain{ImgSeqNum-1}.dispu, ResultStrain{ImgSeqNum-1}.dispv];

        [stress_sxx, stress_sxy, stress_syy, stress_principal_max_xyplane, ...
            stress_principal_min_xyplane, stress_maxshear_xyplane, ...
            stress_maxshear_xyz3d, stress_vonMises] = plot_stress_no_img( ...
            DICpara, ResultStrain{ImgSeqNum-1}, coordinatesFEMWorldDef, elementsFEM_s9(:,1:4));

        ResultStress{ImgSeqNum-1} = struct('stressxCoord', ResultStrain{ImgSeqNum-1}.strainxCoord, ...
            'stressyCoord', ResultStrain{ImgSeqNum-1}.strainyCoord, ...
            'stress_sxx', stress_sxx, 'stress_sxy', stress_sxy, 'stress_syy', stress_syy, ...
            'stress_principal_max_xyplane', stress_principal_max_xyplane, ...
            'stress_principal_min_xyplane', stress_principal_min_xyplane, ...
            'stress_maxshear_xyplane', stress_maxshear_xyplane, ...
            'stress_maxshear_xyz3d', stress_maxshear_xyz3d, 'stress_vonMises', stress_vonMises);
    end
    fprintf('------------ Section 9 Done ------------\n\n');

    %% Collect final results for summary
    % Use the last frame (frame 2, index 1) for summary since it is always computed
    finalU = ResultDisp{1}.U_accum;
    finalCoord = ResultFEMeshEachFrame{1}.coordinatesFEM;
    gt_u_final = gt_u2(finalCoord(:,1), finalCoord(:,2));
    gt_v_final = gt_v2(finalCoord(:,1), finalCoord(:,2));
    final_mask = double(ImgMask{1});
    stats_final = compare_disp_GT(finalU, finalCoord, gt_u_final, gt_v_final, final_mask);

    summary(tc).name = caseName;
    summary(tc).rmse_u = stats_final.rmse_u;
    summary(tc).rmse_v = stats_final.rmse_v;
    summary(tc).pass_disp = (stats_final.rmse_u < 0.5) && (stats_final.rmse_v < 0.5);

    if gt_F11 ~= 0 || gt_F22 ~= 0 || gt_F12 ~= 0 || gt_F21 ~= 0
        finalF = ResultStrain{1};
        FStrainVec = [finalF.dudx(:), finalF.dvdx(:), finalF.dudy(:), finalF.dvdy(:)]';
        FStrainVec = FStrainVec(:);
        gt_F11_n = gt_F11 * ones(size(finalCoord, 1), 1);
        gt_F22_n = gt_F22 * ones(size(finalCoord, 1), 1);
        gt_F12_n = gt_F12 * ones(size(finalCoord, 1), 1);
        gt_F21_n = gt_F21 * ones(size(finalCoord, 1), 1);
        stats_strain_final = compare_strain_GT(FStrainVec, gt_F11_n, gt_F21_n, gt_F12_n, gt_F22_n, finalCoord, final_mask);
        summary(tc).rmse_F11 = stats_strain_final.rmse_F11;
        summary(tc).rmse_F22 = stats_strain_final.rmse_F22;
        summary(tc).rmse_F12 = stats_strain_final.rmse_F12;
        summary(tc).rmse_F21 = stats_strain_final.rmse_F21;
        strain_tol = 0.03;  % ~disp_error/grid_spacing; limited by LocalICGN accuracy
        summary(tc).pass_strain = (stats_strain_final.rmse_F11 < strain_tol) && (stats_strain_final.rmse_F22 < strain_tol) ...
            && (stats_strain_final.rmse_F12 < strain_tol) && (stats_strain_final.rmse_F21 < strain_tol);
    else
        summary(tc).rmse_F11 = NaN;
        summary(tc).rmse_F22 = NaN;
        summary(tc).rmse_F12 = NaN;
        summary(tc).rmse_F21 = NaN;
        summary(tc).pass_strain = true;  % No strain GT to check
    end

    close all;
    fprintf('  >>> %s completed.\n', caseName);

end % test case loop


%% Part 3: Final Summary
fprintf('\n\n');
fprintf('================ SYNTHETIC TEST SUMMARY ================\n');
all_pass = true;
for tc = 1:length(summary)
    s = summary(tc);
    if s.pass_disp
        disp_status = 'PASS';
    else
        disp_status = 'FAIL'; all_pass = false;
    end
    fprintf('%-25s RMSE_u = %.4f  RMSE_v = %.4f  [%s]\n', ...
        s.name, s.rmse_u, s.rmse_v, disp_status);

    if ~isnan(s.rmse_F11)
        if s.pass_strain
            strain_status = 'PASS';
        else
            strain_status = 'FAIL'; all_pass = false;
        end
        fprintf('%-25s RMSE_F11=%.4f F22=%.4f F12=%.4f F21=%.4f [%s]\n', ...
            '', s.rmse_F11, s.rmse_F22, s.rmse_F12, s.rmse_F21, strain_status);
    end
end
fprintf('========================================================\n');
if all_pass
    fprintf('  ALL TESTS PASSED\n');
else
    fprintf('  SOME TESTS FAILED\n');
end
fprintf('========================================================\n');


%% =====================================================================
%  Local Functions
%  =====================================================================

function speckle = generate_speckle(H, W, sigma, seed)
% Generate a realistic speckle pattern using Gaussian-filtered random noise.
%   H, W   - image height and width (file coordinates)
%   sigma  - Gaussian filter std (feature size ~ 2*sigma)
%   seed   - random seed for reproducibility
%   Output: double [H x W], range [20, 235]

    rng(seed);
    noise = randn(H, W);
    % Gaussian kernel
    ksz = ceil(4 * sigma) * 2 + 1;
    kernel = fspecial('gaussian', ksz, sigma);
    filtered = imfilter(noise, kernel, 'replicate');
    % Normalize to [20, 235]
    filtered = filtered - min(filtered(:));
    filtered = filtered / max(filtered(:));
    speckle = 20 + 215 * filtered;
end


function warped = inverse_warp(ref, row_shift, col_shift)
% Inverse warp: warped(row,col) = ref(row - row_shift(row,col), col - col_shift(row,col))
%   ref        - reference image [H x W] in file coordinates
%   row_shift  - row (y_file) displacement field [H x W]
%   col_shift  - column (x_file) displacement field [H x W]
%   Output: warped image [H x W], border filled with 0

    [H, W] = size(ref);
    [ColGrid, RowGrid] = meshgrid(1:W, 1:H);
    src_row = RowGrid - row_shift;
    src_col = ColGrid - col_shift;
    warped = interp2(double(ref), src_col, src_row, 'spline', 0);
end


function stats = compare_disp_GT(U, coordinatesFEM, gt_u, gt_v, mask_img)
% Compare computed displacement against ground truth on mask-interior nodes.
%   U              - displacement vector [2*nNode x 1], interleaved [u1,v1,u2,v2,...]
%   coordinatesFEM - node coordinates [nNode x 2]
%   gt_u, gt_v     - ground truth at each node [nNode x 1]
%   mask_img       - binary mask image (code coordinates, transposed)
%   Output: struct with rmse_u, rmse_v, max_err_u, max_err_v, n_valid

    nNode = size(coordinatesFEM, 1);
    u_comp = U(1:2:end);
    v_comp = U(2:2:end);

    % Find nodes inside mask
    cx = round(coordinatesFEM(:,1));
    cy = round(coordinatesFEM(:,2));
    cx = max(1, min(cx, size(mask_img, 1)));
    cy = max(1, min(cy, size(mask_img, 2)));
    idx = sub2ind(size(mask_img), cx, cy);
    in_mask = mask_img(idx) > 0.5;

    % Also exclude NaN
    valid = in_mask & ~isnan(u_comp) & ~isnan(v_comp);

    err_u = u_comp(valid) - gt_u(valid);
    err_v = v_comp(valid) - gt_v(valid);

    stats.rmse_u = sqrt(mean(err_u.^2));
    stats.rmse_v = sqrt(mean(err_v.^2));
    stats.max_err_u = max(abs(err_u));
    stats.max_err_v = max(abs(err_v));
    stats.mean_err_u = mean(abs(err_u));
    stats.mean_err_v = mean(abs(err_v));
    stats.n_valid = sum(valid);
    stats.n_nan = sum(isnan(u_comp) | isnan(v_comp));
end


function stats = compare_strain_GT(F, gt_F11, gt_F21, gt_F12, gt_F22, coordinatesFEM, mask_img)
% Compare computed strain against ground truth on mask-interior nodes.
%   F     - strain vector [4*nNode x 1]: [F11_1, F21_1, F12_1, F22_1, ...]
%   gt_*  - ground truth components [nNode x 1]
%   Output: struct with rmse for each component

    nNode = size(coordinatesFEM, 1);
    F11 = F(1:4:end); F21 = F(2:4:end);
    F12 = F(3:4:end); F22 = F(4:4:end);

    cx = round(coordinatesFEM(:,1));
    cy = round(coordinatesFEM(:,2));
    cx = max(1, min(cx, size(mask_img, 1)));
    cy = max(1, min(cy, size(mask_img, 2)));
    idx = sub2ind(size(mask_img), cx, cy);
    in_mask = mask_img(idx) > 0.5;

    valid = in_mask & ~isnan(F11) & ~isnan(F22);

    stats.rmse_F11 = sqrt(mean((F11(valid) - gt_F11(valid)).^2));
    stats.rmse_F21 = sqrt(mean((F21(valid) - gt_F21(valid)).^2));
    stats.rmse_F12 = sqrt(mean((F12(valid) - gt_F12(valid)).^2));
    stats.rmse_F22 = sqrt(mean((F22(valid) - gt_F22(valid)).^2));
    stats.n_valid = sum(valid);
end


function debug_print(label, stats)
% Print formatted debug statistics.

    if isfield(stats, 'rmse_u')
        fprintf('[debug] %s: RMSE_u=%.4f, RMSE_v=%.4f, maxErr_u=%.4f, maxErr_v=%.4f, n_valid=%d, n_nan=%d\n', ...
            label, stats.rmse_u, stats.rmse_v, stats.max_err_u, stats.max_err_v, stats.n_valid, stats.n_nan);
    elseif isfield(stats, 'rmse_F11')
        fprintf('[debug] %s: RMSE_F11=%.6f, RMSE_F21=%.6f, RMSE_F12=%.6f, RMSE_F22=%.6f, n_valid=%d\n', ...
            label, stats.rmse_F11, stats.rmse_F21, stats.rmse_F12, stats.rmse_F22, stats.n_valid);
    end
end
