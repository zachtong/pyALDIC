% export_matlab_intermediates.m
% Run AL-DIC on a synthetic test case with ExportIntermediates=true
% to generate checkpoint .mat files for Python cross-validation.
%
% Usage: Run from anywhere — the script auto-detects AL-DIC project root.
%   >> run('path/to/scripts/export_matlab_intermediates.m')

close all; clear; clc;

% Auto-detect project root: this script lives in <root>/scripts/
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');
cd(projectRoot);
fprintf('Working directory: %s\n', pwd);

addpath('./config', './io', './mesh', './solver', './strain', './plotting', ...
    './third_party');

% Compile MEX if needed
try
    mex -O -outdir ./third_party ./third_party/ba_interp2_spline.cpp;
catch
end

%% Generate synthetic case3_affine (standard cross-validation case)
H = 256; W = 256;
rng(42);
noise = randn(H, W);
ksz = ceil(4*3)*2 + 1;
kernel = fspecial('gaussian', ksz, 3);
filtered = imfilter(noise, kernel, 'replicate');
filtered = filtered - min(filtered(:));
filtered = filtered / max(filtered(:));
ref_speckle = 20 + 215 * filtered;

% Displacement: 2% affine expansion
[Xfile, Yfile] = meshgrid(1:W, 1:H);
u2 = 0.02*(Xfile - 128);
v2 = 0.02*(Yfile - 128);
u3 = 0.04*(Xfile - 128);
v3 = 0.04*(Yfile - 128);

% Mask: circular
mask_file = ((Xfile - 128).^2 + (Yfile - 128).^2) <= 90^2;

% Warp
frame2 = interp2(double(ref_speckle), Xfile - u2, Yfile - v2, 'spline', 0);
frame3 = interp2(double(ref_speckle), Xfile - u3, Yfile - v3, 'spline', 0);

%% Setup for pipeline
nFrames = 3;
Img = cell(1, nFrames);
ImgMask = cell(1, nFrames);
Img{1} = ref_speckle';       % Transpose to code coordinates
Img{2} = frame2';
Img{3} = frame3';
ImgMask{1} = logical(mask_file)';
ImgMask{2} = logical(mask_file)';
ImgMask{3} = logical(mask_file)';

file_name = cell(6, nFrames);
for i = 1:nFrames
    file_name{1,i} = sprintf('frame_%02d.tif', i);
    file_name{2,i} = pwd;
end

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
DICpara = dicpara_default(DICpara);

%% Run pipeline with checkpoint export
exportDir = fullfile(pwd, 'tests', 'fixtures', 'matlab_checkpoints');
if ~exist(exportDir, 'dir'), mkdir(exportDir); end

fprintf('\n=== Running pipeline with ExportIntermediates ===\n');
results = run_aldic(DICpara, file_name, Img, ImgMask, ...
    'ExportIntermediates', true, 'ExportDir', exportDir);

fprintf('\n=== Export complete. Checkpoints saved to: ===\n');
fprintf('  %s\n', exportDir);

% Also save the ground truth and input images for cross-validation
save(fullfile(exportDir, 'ground_truth.mat'), ...
    'ref_speckle', 'mask_file', 'u2', 'v2', 'u3', 'v3', 'DICpara', '-v7');
fprintf('  Saved ground_truth.mat\n');
fprintf('\nDone.\n');
