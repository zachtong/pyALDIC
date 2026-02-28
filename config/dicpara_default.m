function DICpara = dicpara_default(varargin)
%dicpara_default  Return a DICpara struct with all fields set to defaults.
%
%   DICpara = DICpara_default()
%       Returns the default parameter struct.
%
%   DICpara = DICpara_default(DICparaIn)
%       Merges user-supplied fields in DICparaIn into the defaults,
%       so any field the user already set is preserved.
%
% =========================================================================
%  STAQ-DIC Parameter Reference
% =========================================================================
%
%  This file is the SINGLE SOURCE OF TRUTH for every DICpara field.
%  All parameters are grouped by function and documented inline.
%
%  Original AL-DIC by Jin Yang (Caltech). Refactored 2026.
% =========================================================================

    %% ====================================================================
    %  1. IMAGE LOADING & ROI
    % =====================================================================

    % Method used to load images (0=folder, 1=prefix, 2=manual select).
    DICpara.LoadImgMethod = 0;

    % Region of Interest. Struct with sub-fields:
    %   .gridx = [xmin, xmax]   (row pixel range)
    %   .gridy = [ymin, ymax]   (column pixel range)
    % Set interactively by ReadImage / ReadImageQuadtree if left empty.
    DICpara.gridxyROIRange = struct('gridx', [], 'gridy', []);

    % Image size [rows, cols]. Populated automatically after loading.
    DICpara.ImgSize = [];

    % Problem dimensionality (always 2 for 2D-DIC).
    DICpara.DIM = 2;

    % Image bit depth (auto-detected from image metadata).
    DICpara.imgBitDepth = 8;

    % Binary mask for the reference image (ones = valid). [] = no mask.
    DICpara.ImgRefMask = [];

    % Whether to use image masks (true/false).
    DICpara.useMasks = false;

    %% ====================================================================
    %  2. MESH & SUBSET PARAMETERS
    % =====================================================================

    % DIC subset (window) size in pixels. Scalar or [wx, wy].
    DICpara.winsize = 40;

    % Subset step size in pixels (spacing between grid nodes). Scalar or [sx, sy].
    DICpara.winstepsize = 20;

    % Finest element size for adaptive quadtree mesh. Only used when
    % meshType = 'quadtree'.
    DICpara.winsizeMin = 8;

    % Mesh type: 'uniform' (regular rectangular grid) or 'quadtree'
    % (adaptive refinement based on image features).
    DICpara.meshType = 'uniform';

    % Per-node window size list [Nx2] for adaptive subset sizing in
    % quadtree ADMM iterations. [] = use uniform winsize for all nodes.
    DICpara.winsize_List = [];

    %% ====================================================================
    %  3. IMAGE SEQUENCE & REFERENCE UPDATE
    % =====================================================================

    % Number of frames between reference image updates (incremental mode).
    % Set to 1 for updating every frame.
    DICpara.ImgSeqIncUnit = 1;

    % Whether to update the ROI when updating the reference image.
    % 0 = keep original ROI, 1 = re-select ROI.
    DICpara.ImgSeqIncROIUpdateOrNot = 0;

    %% ====================================================================
    %  4. FFT INITIAL GUESS (INTEGER SEARCH)
    % =====================================================================

    % Whether to redo FFT cross-correlation for initial guess each frame.
    % 1 = always redo FFT, 0 = reuse previous result when possible.
    DICpara.NewFFTSearch = 1;

    % FFT search method:
    %   0 = multigrid pyramid
    %   1 = whole field
    %   2 = manual seed + interpolation
    DICpara.InitFFTSearchMethod = 1;

    % Size of FFT search region in pixels. Scalar or [size_x, size_y].
    % Must be larger than the expected maximum displacement magnitude.
    DICpara.SizeOfFFTSearchRegion = 120;

    % Cross-correlation threshold to classify points as discontinuous.
    % Points below this threshold get special RBF treatment (quadtree only).
    % Range: [0, 1]. Higher = stricter.
    DICpara.discontinuity_threshold_cc = 0.85;

    % Number of nearest neighbors for discontinuity region mapping.
    DICpara.k_nearest_neighbors = 3;

    %% ====================================================================
    %  5. POD-GPR PREDICTION (for reusing previous results)
    % =====================================================================

    % Enable POD-GPR prediction for initial displacement guess.
    % When false, uses previous frame result directly (if NewFFTSearch=0).
    DICpara.usePODGPR = false;

    % Number of past frames used for POD-GPR displacement prediction.
    % Only used when usePODGPR = true and ImgSeqNum >= POD_startFrame.
    DICpara.POD_nTime = 5;

    % Number of POD basis modes for GPR prediction.
    DICpara.POD_nBasis = 3;

    % Frame number threshold: below this, directly use previous result;
    % above this, use POD-GPR prediction.
    DICpara.POD_startFrame = 7;

    %% ====================================================================
    %  6. IC-GN SOLVER (SUBPROBLEM 1 - LOCAL)
    % =====================================================================

    % IC-GN iteration method: 'GaussNewton' or 'LevenbergMarquardt'.
    DICpara.ICGNmethod = 'GaussNewton';

    % IC-GN local convergence tolerance (norm of update / norm of solution).
    DICpara.tol = 1e-2;

    % Number of parallel workers. 0 = serial; N = use N workers.
    DICpara.ClusterNo = 0;

    %% ====================================================================
    %  7. ADMM / AUGMENTED LAGRANGIAN SOLVER
    % =====================================================================

    % ADMM penalty parameter mu (augmented Lagrangian multiplier weight).
    DICpara.mu = 1e-3;

    % Beta penalty parameter search list. Set to [] to auto-generate from
    % winstepsize and mu.
    %   Auto-generation formula: betaRange .* mean(winstepsize)^2 .* mu
    DICpara.betaRange = [1e-3, 1e-2, 1e-1];

    % Cached optimal beta from previous frame. [] = not yet computed.
    DICpara.beta = [];

    % Maximum number of ADMM iterations (outer loop).
    DICpara.ADMM_maxIter = 3;

    % ADMM global convergence tolerance (norm of inter-step update).
    DICpara.ADMM_tol = 1e-2;

    % Gauss quadrature integration order for FEM strain computation.
    DICpara.GaussPtOrder = 2;

    % Regularization parameter alpha in Subproblem 2 FEM solver.
    DICpara.alpha = 0;

    % Debug flag: skip global ADMM step (Sections 5-6) if set to false.
    % true = normal operation, false = local-only (for debugging).
    DICpara.UseGlobalStep = true;

    %% ====================================================================
    %  8. SUBPROBLEM 2 SOLVER
    % =====================================================================

    % Solver for Subproblem 2:
    %   1 = Finite Difference (FD)
    %   2 = Finite Element Method (FEM)
    % Note: Quadtree meshes always use FEM regardless of this setting.
    DICpara.Subpb2FDOrFEM = 2;

    %% ====================================================================
    %  9. SMOOTHING & FILTERING
    % =====================================================================

    % Gaussian filter size for displacement smoothing (0 = no filter).
    DICpara.DispFilterSize = 0;

    % Gaussian filter standard deviation for displacement smoothing.
    DICpara.DispFilterStd = 0;

    % Gaussian filter size for strain smoothing (0 = no filter).
    DICpara.StrainFilterSize = 0;

    % Gaussian filter standard deviation for strain smoothing.
    DICpara.StrainFilterStd = 0;

    % RBF regularization smoothness for displacement (quadtree only, 0 = none).
    DICpara.DispSmoothness = 0;

    % RBF regularization smoothness for strain (quadtree only, 0 = none).
    DICpara.StrainSmoothness = 0;

    % Whether to apply additional smoothing in post-processing (Section 8).
    % 0 = yes, 1 = no.
    DICpara.DoYouWantToSmoothOnceMore = 1;

    % Regularization smoothness for Section 8 strain computation (quadtree).
    DICpara.smoothness = 0;

    %% ====================================================================
    %  10. STRAIN COMPUTATION
    % =====================================================================

    % Strain computation method:
    %   0 = direct from ALDIC deformation gradient
    %   1 = central finite difference
    %   2 = plane fitting (local least-squares)
    %   3 = finite element method
    DICpara.MethodToComputeStrain = 2;

    % Search radius (px) for plane fitting strain method (MethodToComputeStrain=2).
    % Must be larger than the maximum node spacing (winsizeMin) to include >=4 neighbors.
    % Typical: 2-3x winsizeMin (e.g. 20 for winsizeMin=8).
    DICpara.StrainPlaneFitRad = 20;

    % Strain type:
    %   0 = infinitesimal (engineering) strain
    %   1 = Eulerian-Almansi strain
    %   2 = Green-Lagrangian strain
    %   3 = other
    DICpara.StrainType = 0;

    %% ====================================================================
    %  11. STRESS COMPUTATION
    % =====================================================================

    % Material model:
    %   1 = plane stress (linear elastic)
    %   2 = plane strain (linear elastic)
    %   3 = other / custom
    DICpara.MaterialModel = 1;

    % Material parameters (used when MaterialModel = 1 or 2).
    DICpara.MaterialModelPara.YoungsModulus = [];    % Pa, e.g. 69e9
    DICpara.MaterialModelPara.PoissonsRatio = [];    % dimensionless, e.g. 0.3

    %% ====================================================================
    %  12. VISUALIZATION & OUTPUT
    % =====================================================================

    % Unit conversion: physical units per pixel (e.g., um/px).
    DICpara.um2px = 1;

    % Which image to overlay results on:
    %   0 = first (reference) image
    %   1 = current deformed image
    DICpara.Image2PlotResults = 1;

    % Figure save format:
    %   1 = jpg
    %   2 = pdf
    %   3 = png
    %   4 = fig (MATLAB figure)
    DICpara.MethodToSaveFig = 1;

    % Transparency for overlaying results on original images.
    %   1   = no overlay (pure color plot)
    %   0   = fully opaque original image
    %   0.5 = 50% blend
    DICpara.OrigDICImgTransparency = 1;

    % Output directory path for saving figures. [] = prompt user.
    DICpara.outputFilePath = [];

    % Reference frame mode:
    %   'incremental'   = each frame uses previous frame as reference
    %   'accumulative'  = always use first frame as reference
    DICpara.referenceMode = 'incremental';

    %% ====================================================================
    %  MERGE USER-SUPPLIED FIELDS
    % =====================================================================
    if nargin >= 1 && isstruct(varargin{1})
        userPara = varargin{1};
        fields = fieldnames(userPara);
        for i = 1:length(fields)
            DICpara.(fields{i}) = userPara.(fields{i});
        end
    end

end
