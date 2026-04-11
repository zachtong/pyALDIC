% =========================================================================
% test_aldic_synthetic.m
% Synthetic test data & validation script for STAQ-DIC pipeline.
%
% Generates Gaussian speckle images with known displacement fields,
% runs the full ALDIC pipeline via run_aldic(), and validates results
% against ground truth.
%
% Ten test cases:
%   1. Zero displacement
%   2. Uniform translation (u=2.5, v=-1.8)
%   3. Affine deformation (2% uniform expansion)
%   4. Annular mask with affine deformation (ring mask, winsizeMin=4)
%   5. Simple shear (du/dy = 0.015)
%   6. Large deformation (10% stretch + 5% shear)
%   7. Multi-frame incremental (1 px/frame translation, 3 frames)
%   8. Multi-frame accumulative (same as 7, accumulative mode)
%   9. Local-only DIC (UseGlobalStep=false, compare with case3)
%  10. Pure rotation (2-degree rigid rotation)
%
% Usage: Run this script in MATLAB from the STAQ-DIC-GUI root directory.
% =========================================================================

close all; clear; clc; clearvars -global;
fprintf('============================================================\n');
fprintf('  STAQ-DIC Synthetic Test Suite\n');
fprintf('============================================================\n\n');

%% Part 0: Configuration
addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
    './third_party');

% Compile MEX if needed
try
    mex -O -outdir ./third_party ./third_party/ba_interp2_spline.cpp;
catch
end

H = 256; W = 256;

% Define test cases: {name, frame2_u_func, frame2_v_func, frame3_u_func, frame3_v_func,
%                      GT_strain_F11, GT_strain_F22, mask_type, DICpara_overrides,
%                      GT_strain_F12, GT_strain_F21, disp_tol, strain_tol}
test_cases = {
    'case1_zero',        @(x,y) zeros(size(x)), @(x,y) zeros(size(x)), ...
                         @(x,y) zeros(size(x)), @(x,y) zeros(size(x)), ...
                         0, 0, 'solid', struct(), 0, 0, 0.5, 0.03;
    'case2_translation', @(x,y)  2.5*ones(size(x)), @(x,y) -1.8*ones(size(x)), ...
                         @(x,y)  5.0*ones(size(x)), @(x,y) -3.6*ones(size(x)), ...
                         0, 0, 'solid', struct(), 0, 0, 0.5, 0.03;
    'case3_affine',      @(x,y) 0.02*(x-128), @(x,y) 0.02*(y-128), ...
                         @(x,y) 0.04*(x-128), @(x,y) 0.04*(y-128), ...
                         0.02, 0.02, 'solid', struct(), 0, 0, 0.5, 0.03;
    'case4_annular',     @(x,y) 0.02*(x-128), @(x,y) 0.02*(y-128), ...
                         @(x,y) 0.04*(x-128), @(x,y) 0.04*(y-128), ...
                         0.02, 0.02, 'annular', struct('winsizeMin', 4), 0, 0, 0.5, 0.03;
    'case5_shear',       @(x,y) 0.015*(y-128), @(x,y) zeros(size(x)), ...
                         @(x,y) 0.030*(y-128), @(x,y) zeros(size(x)), ...
                         0, 0, 'solid', struct(), 0.015, 0, 0.5, 0.03;
    'case6_large_deform', @(x,y) 0.10*(x-128)+0.05*(y-128), @(x,y) 0.05*(x-128)+0.10*(y-128), ...
                          @(x,y) 0.20*(x-128)+0.10*(y-128), @(x,y) 0.10*(x-128)+0.20*(y-128), ...
                          0.10, 0.10, 'solid', struct('winsize',48,'SizeOfFFTSearchRegion',25), 0.05, 0.05, 1.0, 0.05;
    'case7_multiframe_incr', @(x,y) 1.0*ones(size(x)), @(x,y) zeros(size(x)), ...
                              @(x,y) 2.0*ones(size(x)), @(x,y) zeros(size(x)), ...
                              0, 0, 'solid', struct('referenceMode','incremental'), 0, 0, 0.5, 0.03;
    'case8_multiframe_accum', @(x,y) 1.0*ones(size(x)), @(x,y) zeros(size(x)), ...
                               @(x,y) 2.0*ones(size(x)), @(x,y) zeros(size(x)), ...
                               0, 0, 'solid', struct(), 0, 0, 0.5, 0.03;
    'case9_local_only',  @(x,y) 0.02*(x-128), @(x,y) 0.02*(y-128), ...
                         @(x,y) 0.04*(x-128), @(x,y) 0.04*(y-128), ...
                         0.02, 0.02, 'solid', struct('UseGlobalStep',false), 0, 0, 0.5, 0.03;
    'case10_rotation',   @(x,y) (x-128)*(cos(pi/90)-1)-(y-128)*sin(pi/90), ...
                         @(x,y) (x-128)*sin(pi/90)+(y-128)*(cos(pi/90)-1), ...
                         @(x,y) (x-128)*(cos(pi/45)-1)-(y-128)*sin(pi/45), ...
                         @(x,y) (x-128)*sin(pi/45)+(y-128)*(cos(pi/45)-1), ...
                         cos(pi/90)-1, cos(pi/90)-1, 'solid', struct(), -sin(pi/90), sin(pi/90), 0.5, 0.03;
};

% Masks (in file coordinates: rows=y_file, cols=x_file)
% After transpose to code coords: dim1=x_code, dim2=y_code
[Xfile, Yfile] = meshgrid(1:W, 1:H);  % Xfile=col=x_file, Yfile=row=y_file
mask_solid = ((Xfile - 128).^2 + (Yfile - 128).^2) <= 90^2;  % [H x W] in file coords
mask_annular = mask_solid & ~(((Xfile - 128).^2 + (Yfile - 128).^2) <= 40^2);  % ring

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
        u_code_at_file = u_func(Xfile, Yfile);
        v_code_at_file = v_func(Xfile, Yfile);

        % inverse_warp: warped(row,col) = ref(row - row_shift, col - col_shift)
        warped_file = inverse_warp(ref_speckle_file, v_code_at_file, u_code_at_file);

        imwrite(uint8(warped_file), fullfile(imgDir, sprintf('frame_%02d.tif', frame_idx)));
        imwrite(uint8(255 * double(mask_file)), fullfile(maskDir, sprintf('mask_%02d.tif', frame_idx)));
    end

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

    % ====== Load images and masks ======
    nFrames = 3;
    Img = cell(1, nFrames);
    ImgMask = cell(1, nFrames);

    for i = 1:nFrames
        raw = double(imread(fullfile(imgDir, sprintf('frame_%02d.tif', i))));
        Img{i} = raw';  % Transpose: file [H x W] -> code [W x H]

        rawMask = imread(fullfile(maskDir, sprintf('mask_%02d.tif', i)));
        ImgMask{i} = logical(rawMask)';  % Transpose
    end

    % Build file_name cell: row1=filename, row2=folder
    file_name = cell(6, nFrames);
    for i = 1:nFrames
        file_name{1, i} = sprintf('frame_%02d.tif', i);
        file_name{2, i} = imgDir;
    end

    % ====== Build DICpara ======
    DICpara = struct();
    DICpara.winsize = 32;
    DICpara.winstepsize = 16;
    DICpara.winsizeMin = 8;
    DICpara.ImgSize = size(Img{1});
    DICpara.gridxyROIRange.gridx = [1, 256];
    DICpara.gridxyROIRange.gridy = [1, 256];
    DICpara.NewFFTSearch = 1;
    DICpara.SizeOfFFTSearchRegion = 10;
    DICpara.referenceMode = 'accumulative';
    DICpara.ADMM_maxIter = 3;
    DICpara.MethodToComputeStrain = 2;
    DICpara.StrainPlaneFitRad = 20;
    DICpara.showPlots = false;

    % Apply per-case DICpara overrides
    overrides = test_cases{tc, 9};
    fields = fieldnames(overrides);
    for i = 1:length(fields)
        DICpara.(fields{i}) = overrides.(fields{i});
    end
    DICpara = dicpara_default(DICpara);

    % ====== Run pipeline ======
    results = run_aldic(DICpara, file_name, Img, ImgMask);

    % ====== Validate results ======
    disp_tol = test_cases{tc, 12};
    strain_tol = test_cases{tc, 13};

    % --- Frame 2 (index 1) validation ---
    finalU = results.ResultDisp{1}.U_accum;
    finalCoord = results.ResultFEMeshEachFrame{1}.coordinatesFEM;
    final_mask = double(ImgMask{1});

    gt_u_final = gt_u2(finalCoord(:,1), finalCoord(:,2));
    gt_v_final = gt_v2(finalCoord(:,1), finalCoord(:,2));
    stats_final = compare_disp_GT(finalU, finalCoord, gt_u_final, gt_v_final, final_mask);
    debug_print(sprintf('%s frame2 displacement', caseName), stats_final);

    summary(tc).name = caseName;
    summary(tc).rmse_u = stats_final.rmse_u;
    summary(tc).rmse_v = stats_final.rmse_v;
    summary(tc).pass_disp = (stats_final.rmse_u < disp_tol) && (stats_final.rmse_v < disp_tol);

    % --- Frame 3 (last frame) cumulative displacement validation ---
    if length(results.ResultDisp) >= 2
        finalU3 = results.ResultDisp{2}.U_accum;
        gt_u3_final = gt_u3(finalCoord(:,1), finalCoord(:,2));
        gt_v3_final = gt_v3(finalCoord(:,1), finalCoord(:,2));
        stats_f3 = compare_disp_GT(finalU3, finalCoord, gt_u3_final, gt_v3_final, final_mask);
        debug_print(sprintf('%s frame3 displacement', caseName), stats_f3);
        summary(tc).rmse_u_f3 = stats_f3.rmse_u;
        summary(tc).rmse_v_f3 = stats_f3.rmse_v;
        summary(tc).pass_disp_f3 = (stats_f3.rmse_u < disp_tol) && (stats_f3.rmse_v < disp_tol);
        summary(tc).pass_disp = summary(tc).pass_disp && summary(tc).pass_disp_f3;
    else
        summary(tc).rmse_u_f3 = NaN;
        summary(tc).rmse_v_f3 = NaN;
        summary(tc).pass_disp_f3 = true;
    end

    % --- Strain validation (frame 2) ---
    if gt_F11 ~= 0 || gt_F22 ~= 0 || gt_F12 ~= 0 || gt_F21 ~= 0
        FStrainVec = [results.ResultStrain{1}.dudx(:), results.ResultStrain{1}.dvdx(:), ...
                      results.ResultStrain{1}.dudy(:), results.ResultStrain{1}.dvdy(:)]';
        FStrainVec = FStrainVec(:);
        nNode = size(finalCoord, 1);
        gt_F11_n = gt_F11 * ones(nNode, 1);
        gt_F22_n = gt_F22 * ones(nNode, 1);
        gt_F12_n = gt_F12 * ones(nNode, 1);
        gt_F21_n = gt_F21 * ones(nNode, 1);
        stats_strain_final = compare_strain_GT(FStrainVec, gt_F11_n, gt_F21_n, gt_F12_n, gt_F22_n, finalCoord, final_mask);
        debug_print(sprintf('%s strain', caseName), stats_strain_final);
        summary(tc).rmse_F11 = stats_strain_final.rmse_F11;
        summary(tc).rmse_F22 = stats_strain_final.rmse_F22;
        summary(tc).rmse_F12 = stats_strain_final.rmse_F12;
        summary(tc).rmse_F21 = stats_strain_final.rmse_F21;
        summary(tc).pass_strain = (stats_strain_final.rmse_F11 < strain_tol) && (stats_strain_final.rmse_F22 < strain_tol) ...
            && (stats_strain_final.rmse_F12 < strain_tol) && (stats_strain_final.rmse_F21 < strain_tol);
    else
        summary(tc).rmse_F11 = NaN;
        summary(tc).rmse_F22 = NaN;
        summary(tc).rmse_F12 = NaN;
        summary(tc).rmse_F21 = NaN;
        summary(tc).pass_strain = true;
    end

    fprintf('  >>> %s completed.\n', caseName);
end


%% Part 3: Final Summary
fprintf('\n\n');
fprintf('==================== SYNTHETIC TEST SUMMARY ====================\n');
all_pass = true;
for tc = 1:length(summary)
    s = summary(tc);
    dtol = test_cases{tc, 12};
    stol = test_cases{tc, 13};

    if s.pass_disp
        disp_status = 'PASS';
    else
        disp_status = 'FAIL'; all_pass = false;
    end
    fprintf('%-25s F2: RMSE_u=%.4f RMSE_v=%.4f (tol=%.1f) [%s]\n', ...
        s.name, s.rmse_u, s.rmse_v, dtol, disp_status);

    % Show frame 3 results if available
    if ~isnan(s.rmse_u_f3)
        if s.pass_disp_f3
            f3_status = 'PASS';
        else
            f3_status = 'FAIL';
        end
        fprintf('%-25s F3: RMSE_u=%.4f RMSE_v=%.4f [%s]\n', ...
            '', s.rmse_u_f3, s.rmse_v_f3, f3_status);
    end

    if ~isnan(s.rmse_F11)
        if s.pass_strain
            strain_status = 'PASS';
        else
            strain_status = 'FAIL'; all_pass = false;
        end
        fprintf('%-25s Strain: F11=%.5f F22=%.5f F12=%.5f F21=%.5f (tol=%.2f) [%s]\n', ...
            '', s.rmse_F11, s.rmse_F22, s.rmse_F12, s.rmse_F21, stol, strain_status);
    end
end
fprintf('================================================================\n');
if all_pass
    fprintf('  ALL %d TESTS PASSED\n', length(summary));
else
    fprintf('  SOME TESTS FAILED\n');
end
fprintf('================================================================\n');


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
        fprintf('[result] %s: RMSE_u=%.4f, RMSE_v=%.4f, maxErr_u=%.4f, maxErr_v=%.4f, n_valid=%d, n_nan=%d\n', ...
            label, stats.rmse_u, stats.rmse_v, stats.max_err_u, stats.max_err_v, stats.n_valid, stats.n_nan);
    elseif isfield(stats, 'rmse_F11')
        fprintf('[result] %s: RMSE_F11=%.6f, RMSE_F21=%.6f, RMSE_F12=%.6f, RMSE_F22=%.6f, n_valid=%d\n', ...
            label, stats.rmse_F11, stats.rmse_F21, stats.rmse_F12, stats.rmse_F22, stats.n_valid);
    end
end
