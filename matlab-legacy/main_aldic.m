% ---------------------------------------------
% STAQ-DIC: Augmented Lagrangian Digital Image Correlation
% Adaptive quadtree mesh with RBF-based smoothing/strain computation.
% Supports both incremental and accumulative reference frame modes.
%
% Original author: Jin Yang, PhD @Caltech
% Contact: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Refactored: 2026-02
% ---------------------------------------------

%% Section 1: Clear MATLAB environment & mex set up
close all; clear; clc; clearvars -global
fprintf('------------ Section 1 Start ------------ \n')
setenv('MW_MINGW64_LOC','C:\\TDM-GCC-64');
try
    mex -O -outdir ./third_party ./third_party/ba_interp2_spline.cpp;
catch ME
    errorMessage = sprintf('Error compiling ba_interp2_spline.cpp: %s', ME.message);
    errordlg(errorMessage, 'Compilation Error');
end
addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
        './third_party');
fprintf('------------ Section 1 Done ------------ \n\n')


%% Section 2: Load images and masks (interactive)
fprintf('------------ Section 2 Start ------------ \n')
[file_name, Img, DICpara] = read_images;
DICpara = dicpara_default(DICpara);
disp(['The finest element size in the adaptive quadtree mesh is ', num2str(DICpara.winsizeMin)]);

[~, ImgMask] = read_masks(DICpara);
fprintf('------------ Section 2 Done ------------ \n\n')


%% Run full pipeline
results = run_aldic(DICpara, file_name, Img, ImgMask);


%% Save results
[~, imgname, ~] = fileparts(file_name{1,end});
results_name = ['results_', imgname, '_ws', num2str(DICpara.winsize), ...
    '_st', num2str(DICpara.winstepsize), '.mat'];
save(results_name, '-struct', 'results');
fprintf('Results saved to %s\n', results_name);
