% ---------------------------------------------
% STAQ-DIC: Augmented Lagrangian Digital Image Correlation
% Adaptive quadtree mesh with RBF-based smoothing/strain computation.
% Supports both incremental and accumulative reference frame modes.
%
% Original author: Jin Yang, PhD @Caltech
% Contact: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Refactored: 2026-02 (merged from main_ALDIC_Quadtree_inc.m and _v2.m)
% ---------------------------------------------

%% Section 1: Clear MATLAB environment & mex set up Spline interpolation
close all; clear; clc; clearvars -global
fprintf('------------ Section 1 Start ------------ \n')
setenv('MW_MINGW64_LOC','C:\\TDM-GCC-64');
try
    mex -O ba_interp2_spline.cpp; % mex set up ba_interp2_spline.cpp
catch ME
    errorMessage = sprintf('Error compiling ba_interp2_spline.cpp: %s', ME.message);
    errordlg(errorMessage, 'Compilation Error'); % Displays a pop-up error dialog
end
% [Comment]: If this line reports error but it works before,
% Change line 16 to: "try mex -O ba_interp2_spline.cpp; catch; end"
addpath('./config','./io','./mesh','./solver','./strain','./plotting',...
        './third_party','./third_party/rbfinterp');
fprintf('------------ Section 1 Done ------------ \n\n')


%% Section 2: Load DIC parameters and set up DIC parameters
fprintf('------------ Section 2 Start ------------ \n')

% ====== Read images ======
[file_name,Img,DICpara] = read_images; % Load DIC raw images
DICpara = dicpara_default(DICpara); % Merge with defaults for any missing fields
disp(['The finest element size in the adaptive quadtree mesh is ', num2str(DICpara.winsizeMin)]);

% ====== Load mask files ======
[mask_file_name,ImgMask] = read_masks;

% %%%%%% Uncomment lines below to change the DIC computing region (ROI) manually %%%%%%
% DICpara.gridxROIRange = [gridxROIRange1,gridxROIRange2]; DICpara.gridyROIRange = [Val1, Val2];
% E.g., gridxROIRange = [224,918]; gridyROIRange = [787,1162];
% DICpara.gridxyROIRange.gridx = [10, 410];
% DICpara.gridxyROIRange.gridy = [202, 420];

% ====== Normalize images: fNormalized = (f-f_avg)/(f_std) ======
[ImgNormalized,DICpara.gridxyROIRange] = normalize_img(Img,DICpara.gridxyROIRange);

% ====== Initialize variable storage ======
ResultDisp = cell(length(ImgNormalized)-1,1);    ResultDefGrad = cell(length(ImgNormalized)-1,1);
ResultStrain = cell(length(ImgNormalized)-1,1);  ResultStress = cell(length(ImgNormalized)-1,1);
ResultFEMeshEachFrame = cell(length(ImgNormalized)-1,1); % To store FE-mesh for each frame: needs future improvment to make it more efficient.
ResultFEMesh = cell(ceil((length(ImgNormalized)-1)/DICpara.ImgSeqIncUnit),1); % For incremental DIC mode
% DICpara.SizeOfFFTSearchRegion already set in dicpara_default
fprintf('------------ Section 2 Done ------------ \n\n')


%% Debug options
UseGlobal = DICpara.UseGlobalStep;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To solve each frame in an image sequence
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ImgSeqNum = 2 : length(ImgNormalized)

    close all;
    disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]); % Report current frame #

    % ====== Load reference image ======
    if strcmp(DICpara.referenceMode, 'accumulative')
        fNormalizedMask = double( ImgMask{1} );           % Always use first frame as reference
        fNormalized = ImgNormalized{1} .* fNormalizedMask;
    else % 'incremental'
        fNormalizedMask = double( ImgMask{ImgSeqNum-1} ); % Use previous frame as reference
        fNormalized = ImgNormalized{ImgSeqNum-1} .* fNormalizedMask;
    end
    Df = img_gradient(fNormalized,fNormalized,fNormalizedMask); % Compute image gradients

    gNormalizedMask = double(ImgMask{ImgSeqNum}); % Load the mask file of current frame
    gNormalized = ImgNormalized{ ImgSeqNum } .* gNormalizedMask ; % Load current deformed image frame

    DICpara.ImgRefMask = fNormalizedMask;

    figure,
    subplot(2,2,1); imshow(fNormalized'); title('fNormalized'); colorbar;
    subplot(2,2,2); imshow(gNormalized'); title('gNormalized'); colorbar;
    subplot(2,2,3); imshow(fNormalizedMask'); title('f mask'); colorbar;
    subplot(2,2,4); imshow(gNormalizedMask'); title('g mask'); colorbar;


    %% Section 3: Compute an initial guess of the unknown displacement field
    fprintf('\n'); fprintf('------------ Section 3 Start ------------ \n')
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to find or update an initial guess of the unknown displacements.
    % The key idea is to either to use a new FFT-based cross correlation peak fitting,
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if ImgSeqNum == 2 || DICpara.NewFFTSearch == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ====== FFT-based cross correlation ======
    DICpara.InitFFTSearchMethod = 1;  % FFT
    [DICpara,x0temp_f,y0temp_f,u_f,v_f,cc] = integer_search(fNormalized,gNormalized,file_name,DICpara);

    % =================== Zach improvement starts 20250624 ================
    xnodes = max([1+0.5*DICpara.winsize,DICpara.gridxyROIRange.gridx(1)])  ...
        : DICpara.winstepsize : min([size(fNormalized,1)-0.5*DICpara.winsize-1,DICpara.gridxyROIRange.gridx(2)]);
    ynodes = max([1+0.5*DICpara.winsize,DICpara.gridxyROIRange.gridy(1)])  ...
        : DICpara.winstepsize : min([size(fNormalized,2)-0.5*DICpara.winsize-1,DICpara.gridxyROIRange.gridy(2)]);

    [x0temp,y0temp] = ndgrid(xnodes,ynodes);

    % A point is only valid if BOTH its u and v displacements are valid numbers.
    valid_indices_u = find(~isnan(u_f(:)));
    valid_indices_v = find(~isnan(v_f(:)));
    valid_indices = intersect(valid_indices_u, valid_indices_v);

    % Define the correlation threshold.
    discontinuity_threshold_cc = DICpara.discontinuity_threshold_cc;

    % This creates a list of local indices relative to the 'valid_indices' vector.
    low_cc_local_indices = find(cc.max(valid_indices) < discontinuity_threshold_cc);

    % Map these local indices back to the original, global indices to get the final list.
    discontinuity_indices = valid_indices(low_cc_local_indices);

    % The smooth indices are all valid points that are not in the discontinuous list.
    smooth_indices = setdiff(valid_indices, discontinuity_indices);

    % This part creates a smooth, continuous field based on high-quality data.
    if ~isempty(smooth_indices)
        % Create the interpolation model using only points from the smooth region.
        op1_smooth = rbfcreate([x0temp_f(smooth_indices), y0temp_f(smooth_indices)]', [u_f(smooth_indices)]', 'RBFFunction', 'thinplate');
        u_smooth = rbfinterp([x0temp(:), y0temp(:)]', op1_smooth);

        op2_smooth = rbfcreate([x0temp_f(smooth_indices), y0temp_f(smooth_indices)]', [v_f(smooth_indices)]', 'RBFFunction', 'thinplate');
        v_smooth = rbfinterp([x0temp(:), y0temp(:)]', op2_smooth);

        % Regularize the smooth field to remove noise.
        u_final = regularizeNd([x0temp(:), y0temp(:)], u_smooth(:), {xnodes', ynodes'}, 1e-3);
        v_final = regularizeNd([x0temp(:), y0temp(:)], v_smooth(:), {xnodes', ynodes'}, 1e-3);
    else
        % The discontinuous region will overwrite parts of this later if it exists.
        u_final = nan(size(x0temp));
        v_final = nan(size(x0temp));
    end

    % This part handles low-correlation points, preserving sharp jumps in displacement.
    if ~isempty(discontinuity_indices)
        % Use knnsearch for a robust nearest-neighbor lookup, avoiding convex hull issues.
        % This is more reliable than scatteredInterpolant for this purpose.
        discontinuous_points_coords = [x0temp_f(discontinuity_indices), y0temp_f(discontinuity_indices)];

        % Process u-displacement
        discontinuous_u_values = u_f(discontinuity_indices);
        nearest_idx_u = knnsearch(discontinuous_points_coords, [x0temp(:), y0temp(:)]);
        u_discontinuous = reshape(discontinuous_u_values(nearest_idx_u), size(x0temp));

        % Process v-displacement
        discontinuous_v_values = v_f(discontinuity_indices);
        nearest_idx_v = knnsearch(discontinuous_points_coords, [x0temp(:), y0temp(:)]);
        v_discontinuous = reshape(discontinuous_v_values(nearest_idx_v), size(x0temp));

        % --- Create a mask to identify where to apply the discontinuous values ---
        % Define the neighborhood size for mapping the discontinuity.
        k_nearest_neighbors = DICpara.k_nearest_neighbors;

        % For each point in the new grid, find the indices of its k nearest neighbors
        [~, nearest_indices_in_orig_grid] = pdist2([x0temp_f(:), y0temp_f(:)], [x0temp(:), y0temp(:)], 'euclidean', 'Smallest', k_nearest_neighbors);

        % A new grid point is considered part of the discontinuous region if any of its
        % k neighbors were flagged as discontinuous.
        is_neighbor_discontinuous = ismember(nearest_indices_in_orig_grid, discontinuity_indices);
        is_discontinuous_on_new_grid = any(is_neighbor_discontinuous, 1);
        is_discontinuous_on_new_grid = reshape(is_discontinuous_on_new_grid, size(x0temp));

        % --- Overwrite the discontinuous regions in the final field ---
        u_final(is_discontinuous_on_new_grid) = u_discontinuous(is_discontinuous_on_new_grid);
        v_final(is_discontinuous_on_new_grid) = v_discontinuous(is_discontinuous_on_new_grid);
    end

    % Use the combined u_final and v_final for subsequent Init and mesh_setup steps.
    u = u_final;
    v = v_final;
    % =================== Zach improvement ends ===================


    % ====== DIC uniform FE-mesh set up ======
    [DICmesh] = mesh_setup(x0temp,y0temp,DICpara); % clear x0temp y0temp;
    % ====== Initial Value ======
    U0 = init_disp(u,v,cc.max,DICmesh.x0,DICmesh.y0,0);

    % Zach Modified
    % Set zero at holes
    linearIndices1 = sub2ind(size(fNormalizedMask), DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
    MaskOrNot1 = fNormalizedMask(linearIndices1);
    nanIndex = find(MaskOrNot1<1);
    U0(2*nanIndex) = nan;
    U0(2*nanIndex-1) = nan;


    % ====== Deal with incremental mode ======
    % %%%%% Old codes %%%%%
    fNormalizedNewIndex = ImgSeqNum-mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit)-1;
    if DICpara.ImgSeqIncUnit == 1, fNormalizedNewIndex = fNormalizedNewIndex-1; end
    ResultFEMesh{1+floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)} = ... % To save first mesh info
        struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM, ...
        'winsize',DICpara.winsize,'winstepsize',DICpara.winstepsize,'gridxyROIRange',DICpara.gridxyROIRange );

    % ====== Generate a quadtree mesh considering sample's complex geometry ======
    DICmesh.elementMinSize = DICpara.winsizeMin; % min element size in the refined quadtree mesh

    % Notes:
    % Hanging nodes and sub-elements are placed on the last
    % All the void regions are generating nodes but we can ignore them
    % using maskfile later.
    [DICmesh, U0] = generate_mesh(DICmesh, DICpara, Df, U0); % Generate the quadtree mesh
    plot_disp_show(U0,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'EdgeColor');

    % ====== Store current mesh ======
    ResultFEMeshEachFrame{ImgSeqNum-1} = struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM,'markCoordHoleEdge',DICmesh.markCoordHoleEdge );

    else
    % ====== Reuse previous mesh, predict U0 from previous results ======
    if DICpara.usePODGPR && ImgSeqNum >= DICpara.POD_startFrame
        % POD-GPR prediction
        nTime = DICpara.POD_nTime;
        np = length(ResultDisp{ImgSeqNum-2}.U)/2;
        T_data_u = zeros(nTime,np); T_data_v = zeros(nTime,np);
        for tempi = 1:nTime
            T_data_u(tempi,:) = ResultDisp{ImgSeqNum-(2+nTime)+tempi, 1}.U(1:2:np*2)';
            T_data_v(tempi,:) = ResultDisp{ImgSeqNum-(2+nTime)+tempi, 1}.U(2:2:np*2)';
        end
        nB = DICpara.POD_nBasis;
        t_train = (ImgSeqNum-1-nTime:ImgSeqNum-2)';
        t_pre = (ImgSeqNum-1)';
        [u_pred,~,~,~] = por_gpr(T_data_u,t_train,t_pre,nB);
        [v_pred,~,~,~] = por_gpr(T_data_v,t_train,t_pre,nB);
        tempu = u_pred(1,:); tempv = v_pred(1,:);
        U0 = [tempu(:),tempv(:)]'; U0 = U0(:);
        disp('POD-GPR prediction used for initial guess.');
    else
        % Use previous frame result as initial guess
        U0 = ResultDisp{ImgSeqNum-2}.U;
        disp('Previous frame result used as initial guess.');
    end
    ResultFEMeshEachFrame{ImgSeqNum-1} = struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM,'markCoordHoleEdge',DICmesh.markCoordHoleEdge );

    end % if ImgSeqNum == 2 || DICpara.NewFFTSearch == 1

    fprintf('------------ Section 3 Done ------------ \n\n')


    %% Section 4: ALDIC Subproblem 1 -or- Local ICGN Subset DIC
    fprintf('------------ Section 4 Start ------------ \n')
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to solve the first local step in ALDIC: Subproblem 1
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % ====== ALStep 1 Subproblem1: Local Subset DIC ======
    mu=0; beta=0; tol=DICpara.tol; ALSolveStep=1; ALSub1Time=zeros(6,1); ALSub2Time=zeros(6,1);
    ConvItPerEle=zeros(size(DICmesh.coordinatesFEM,1),6); ALSub1BadPtNum=zeros(6,1);
    disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****'])
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ------ Start Local DIC IC-GN iteration ------
    [USubpb1,FSubpb1,HtempPar,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp,markCoordHoleStrain] = ...
        local_icgn(U0,DICmesh.coordinatesFEM,Df,fNormalized,gNormalized,DICpara,DICpara.ICGNmethod,tol);
    ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp; ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp; toc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % ====== Thin-plate interpolate bad points =====
    coordinatesFEM = DICmesh.coordinatesFEM;
    U = USubpb1; F = FSubpb1;


    USubpb1World = USubpb1; USubpb1World(2:2:end) = -USubpb1(2:2:end); FSubpb1World = FSubpb1;

    plot_disp_show(USubpb1World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'EdgeColor');
    plot_strain_show(FSubpb1World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'EdgeColor');
    save(['Subpb1_step',num2str(ALSolveStep)],'USubpb1','FSubpb1');
    fprintf('------------ Section 4 Done ------------ \n\n')


    if UseGlobal
        %% Section 5: Subproblem 2 -- solve the global compatible displacement field
        fprintf('------------ Section 5 Start ------------ \n'); tic;
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This section is to solve the global step in ALDIC Subproblem 2
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % ======= ALStep 1 Subproblem 2: Global constraint =======
        % ------ Smooth displacements for a better F ------
        LevelNo=1;  % Smoothing parameters already set in dicpara_default
        if DICpara.DispSmoothness>1e-6, USubpb1 = smooth_disp_rbf(USubpb1,DICmesh,DICpara); end
        if DICpara.StrainSmoothness>1e-6, FSubpb1 = smooth_strain_rbf(FSubpb1,DICmesh,DICpara); end

        % ====== Define penalty parameter ======
        mu = DICpara.mu; udual = 0*FSubpb1; vdual = 0*USubpb1;
        betaList = DICpara.betaRange * mean(DICpara.winstepsize).^2 .* mu; % Tune beta in the betaList
        Err1 = zeros(length(betaList),1); Err2 = Err1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****']);
        alpha = DICpara.alpha;  % No regularization added
        % ====== Solver using finite element method ======
        if ImgSeqNum == 2
            for tempk = 1:length(betaList)
                beta = betaList(tempk); display(['Try #',num2str(tempk),' beta = ',num2str(beta)]);
                alpha=0; [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize),0);
                FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);

                Err1(tempk) = norm(USubpb1-USubpb2,2);
                Err2(tempk) = norm(FSubpb1-FSubpb2,2);
            end

            Err1Norm = (Err1-mean(Err1))/std(Err1); % figure, plot(Err1Norm);
            Err2Norm = (Err2-mean(Err2))/std(Err2); % figure, plot(Err2Norm);
            ErrSum = Err1Norm+Err2Norm; % figure, plot(ErrSum); title('Tune the best \\beta in the subproblem 2');
            [~,indexOfbeta] = min(ErrSum);

            try % Tune the best beta by a quadratic polynomial 0fitting
                [fitobj] = fit(log10(betaList(indexOfbeta-1:1:indexOfbeta+1))',ErrSum(indexOfbeta-1:1:indexOfbeta+1),'poly2');
                p = coeffvalues(fitobj); beta = 10^(-p(2)/2/p(1));
            catch, beta = betaList(indexOfbeta);
            end
            display(['Best beta = ',num2str(beta)]);
        else
            if ~isempty(DICpara.beta)
                beta = DICpara.beta;
            else
                beta = 1e-3*mean(DICpara.winstepsize).^2.*mu;
            end
        end

        % Using the optimal beta to solve the ALDIC Subproblem 2 again
        if abs(beta-betaList(end))>abs(eps)
            [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize),0);
            FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);
            ALSub2Time(ALSolveStep) = toc; toc
        end

        % ------- Smooth strain field --------
        if DICpara.DispSmoothness>1e-6, USubpb2 = smooth_disp_rbf(USubpb2,DICmesh,DICpara); end
        % ------- Don't smooth strain fields near the boundary --------
        for tempk=0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
        if DICpara.StrainSmoothness>1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1,DICmesh,DICpara); end
        for tempk=0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
        for tempk=0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % ------- Save data ------
        save(['Subpb2_step',num2str(ALSolveStep)],'USubpb2','FSubpb2');

        % ======= Update dual variables =======
        udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
        save(['uvdual_step',num2str(ALSolveStep)],'udual','vdual');
        fprintf('------------ Section 5 Done ------------ \n\n')


        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Section 6: ADMM iterations
        fprintf('------------ Section 6 Start ------------ \n')
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This section is the ADMM iteration, where both Subproblems 1 & 2 are solved iteratively.
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % ==================== ADMM AL Loop ==========================
        ALSolveStep = 1; tol2 = DICpara.ADMM_tol; UpdateY = 1e4;
        HPar = cell(21,1); for tempj = 1:21, HPar{tempj} = HtempPar(:,tempj); end

        while (ALSolveStep < DICpara.ADMM_maxIter)
            ALSolveStep = ALSolveStep + 1;  % Update using the last step

            %%%%%%%% These lines can be used to further update each DIC subset window size %%%%%%%
            % Ftemp1 = FSubpb2(1:2:end); Ftemp2 = FSubpb2(2:2:end);
            % [DFtemp1] = global_nodal_strain_rbf(DICmesh,DICpara,Ftemp1);
            % [DFtemp2] = global_nodal_strain_rbf(DICmesh,DICpara,Ftemp2);
            %
            % winsize_x_ub1 = abs(2*FSubpb2(1:4:end)./DFtemp1(1:4:end)); winsize_x_ub2 = abs(2*FSubpb2(3:4:end)./DFtemp1(3:4:end));
            % winsize_y_ub1 = abs(2*FSubpb2(1:4:end)./DFtemp1(2:4:end)); winsize_y_ub2 = abs(2*FSubpb2(3:4:end)./DFtemp1(4:4:end));
            % winsize_x_ub3 = abs(2*FSubpb2(2:4:end)./DFtemp2(1:4:end)); winsize_x_ub4 = abs(2*FSubpb2(4:4:end)./DFtemp2(3:4:end));
            % winsize_y_ub3 = abs(2*FSubpb2(2:4:end)./DFtemp2(2:4:end)); winsize_y_ub4 = abs(2*FSubpb2(4:4:end)./DFtemp2(4:4:end));
            %
            % winsize_x_ub = round(min([winsize_x_ub1,winsize_x_ub2,winsize_x_ub3,winsize_x_ub4,DICpara.winsize*ones(length(winsize_x_ub1),1)],[],2));
            % winsize_x_List = max([winsize_x_ub, 10*ones(length(winsize_x_ub1),1)],[],2);
            % winsize_y_ub = round(min([winsize_y_ub1,winsize_y_ub2,winsize_y_ub3,winsize_y_ub4,DICpara.winsize*ones(length(winsize_y_ub1),1)],[],2));
            % winsize_y_List = max([winsize_y_ub, 10*ones(length(winsize_y_ub1),1)],[],2);
            % winsize_List = 2*ceil([winsize_x_List,winsize_y_List]/2);
            winsize_List = DICpara.winsize*ones(size(DICmesh.coordinatesFEM,1),2);
            DICpara.winsize_List = winsize_List;


            %%%%%%%%%%%%%%%%%%%%%%% Subproblem 1 %%%%%%%%%%%%%%%%%%%%%%%%%
            disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****']);
            tic; [USubpb1,~,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp] = subpb1_solver(...
                USubpb2,FSubpb2,udual,vdual,DICmesh.coordinatesFEM,...
                Df,fNormalized,gNormalized,mu,beta,HPar,ALSolveStep,DICpara,DICpara.ICGNmethod,tol);
            FSubpb1 = FSubpb2; toc
            ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp; ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp;
            save(['Subpb1_step',num2str(ALSolveStep)],'USubpb1','FSubpb1');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ============== Subproblem 2 ==============
            disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****'])
            tic; [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize),0);
            FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);
            ALSub2Time(ALSolveStep) = toc; toc

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ------- Smooth strain field --------
            if DICpara.DispSmoothness>1e-6, USubpb2 = smooth_disp_rbf(USubpb2,DICmesh,DICpara); end
            % ------- Don't change strain fields near the boundary --------
            for tempk=0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
            if DICpara.StrainSmoothness>1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1,DICmesh,DICpara); end
            for tempk=0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
            for tempk=0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end

            save(['Subpb2_step',num2str(ALSolveStep)],'USubpb2','FSubpb2');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Compute norm of UpdateY
            USubpb2_Old = load(['Subpb2_step',num2str(ALSolveStep-1)],'USubpb2');
            USubpb2_New = load(['Subpb2_step',num2str(ALSolveStep)],'USubpb2');
            USubpb1_Old = load(['Subpb1_step',num2str(ALSolveStep-1)],'USubpb1');
            USubpb1_New = load(['Subpb1_step',num2str(ALSolveStep)],'USubpb1');
            if (mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) ~= 0 && (ImgSeqNum>2)) || (ImgSeqNum < DICpara.ImgSeqIncUnit)
                UpdateY = norm((USubpb2_Old.USubpb2 - USubpb2_New.USubpb2), 2)/sqrt(size(USubpb2_Old.USubpb2,1));
                    UpdateY2 = norm((USubpb1_Old.USubpb1 - USubpb1_New.USubpb1), 2)/sqrt(size(USubpb1_Old.USubpb1,1));
                end
            end
            if exist('UpdateY','var'), disp(['Update global step = ',num2str(UpdateY)]); end
            if exist('UpdateY2','var'), disp(['Update local step  = ',num2str(UpdateY2)]); end
            fprintf('*********************************** \n\n');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update dual variables------------------------------
            udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;

            save(['uvdual_step',num2str(ALSolveStep)],'udual','vdual');
            if exist('UpdateY','var') && exist('UpdateY2','var')
                if UpdateY < tol2 || UpdateY2 < tol2
                    break
                end
            end

        end
        fprintf('------------ Section 6 Done ------------ \n\n')
    end

    if UseGlobal
        % Save data
        ResultDisp{ImgSeqNum-1}.U = full(USubpb2);
        ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
        ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb2);
    else
        % Save data
        ResultDisp{ImgSeqNum-1}.U = full(USubpb1);
        ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
        ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb1);
    end

end


%% ------ Plot ------
USubpb2World = USubpb2; USubpb2World(2:2:end) = -USubpb2(2:2:end); FSubpb2World = FSubpb2;
close all; plot_disp_show(USubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'EdgeColor');
plot_strain_show(FSubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'EdgeColor');

% ------ Save results ------
% Find img name and save all the results
[~,imgname,imgext] = fileparts(file_name{1,end});
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame','ALSub1Time','ALSub2Time','ALSolveStep');


%% Section 7: Check convergence
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to check convergence of ADMM
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('------------ Section 7 Start ------------ \n')
% ====== Check convergence ======
fprintf('***** Check convergence ***** \n');
ALSolveStep1 = min(6,ALSolveStep);
disp('==== uhat^(k) - u^(k) ====');
for ALSolveStep = 1:ALSolveStep1
    USubpb2 = load(['Subpb2_step',num2str(ALSolveStep )],'USubpb2');
    USubpb1 = load(['Subpb1_step',num2str(ALSolveStep )],'USubpb1');
    UpdateY = norm((USubpb2.USubpb2 - USubpb1.USubpb1), 2)/sqrt(length(USubpb2.USubpb2));
    disp(num2str(UpdateY));
end
disp('==== Fhat^(k) - F^(k) ====');
for ALSolveStep = 1:ALSolveStep1
    FSubpb1 = load(['Subpb1_step',num2str(ALSolveStep )],'FSubpb1');
    FSubpb2 = load(['Subpb2_step',num2str(ALSolveStep )],'FSubpb2');
    UpdateF = norm((FSubpb1.FSubpb1 - FSubpb2.FSubpb2), 2)/sqrt(length(FSubpb1.FSubpb1));
    disp(num2str(UpdateF));
end
disp('==== uhat^(k) - uhat^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    USubpb2_Old = load(['Subpb2_step',num2str(ALSolveStep-1)],'USubpb2');
    USubpb2_New = load(['Subpb2_step',num2str(ALSolveStep)],'USubpb2');
    UpdateY = norm((USubpb2_Old.USubpb2 - USubpb2_New.USubpb2), 2)/sqrt(length(USubpb2.USubpb2));
    disp(num2str(UpdateY));
end
disp('==== udual^(k) - udual^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    uvdual_Old = load(['uvdual_step',num2str(ALSolveStep-1)],'udual');
    uvdual_New = load(['uvdual_step',num2str(ALSolveStep)],'udual');
    UpdateW = norm((uvdual_Old.udual - uvdual_New.udual), 2)/sqrt(length(uvdual_Old.udual));
    disp(num2str(UpdateW));
end
disp('==== vdual^(k) - vdual^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    uvdual_Old = load(['uvdual_step',num2str(ALSolveStep-1)],'vdual');
    uvdual_New = load(['uvdual_step',num2str(ALSolveStep)],'vdual');
    Updatev = norm((uvdual_Old.vdual - uvdual_New.vdual), 2)/sqrt(length(uvdual_Old.vdual));
    disp(num2str(Updatev));
end
fprintf('------------ Section 7 Done ------------ \n\n')

% ------ clear temp variables ------
clear a ALSub1BadPtNum ALSub1Timetemp atemp b btemp cc ConvItPerEletemp hbar Hbar
clear coordinatesFEMQuadtree elementsFEMQuadtree


%% ====== Transform displacement fields to cumulative ======
if strcmp(DICpara.referenceMode, 'incremental')
    % In incremental mode, RBF-interpolate incremental displacements to cumulative
    tempx = ResultFEMeshEachFrame{1}.coordinatesFEM(:,1);
    tempy = ResultFEMeshEachFrame{1}.coordinatesFEM(:,2);
    coord = [tempx,tempy]; coordCurr = coord;
    hbar = waitbar(0,'Calculate cumulative disp from incremental disp');

    for ImgSeqNum = 2 : length(ImgNormalized)

        waitbar((ImgSeqNum-1)/(size(file_name,2)-1));
        tempx = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM(:,1);
        tempy = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM(:,2);

        tempu = ResultDisp{ImgSeqNum-1}.U(1:2:end);
        tempv = ResultDisp{ImgSeqNum-1}.U(2:2:end);

        op2_x = rbfcreate( [tempx,tempy]',[tempu]','RBFFunction', 'thinplate');
        rbfcheck_maxdiff = rbfcheck(op2_x);
        if rbfcheck_maxdiff > 1e-3, disp('Please check rbf interpolation! Pause here.'); pause; end
        disp_x = rbfinterp([coordCurr(:,1),coordCurr(:,2)]', op2_x );

        op2_y = rbfcreate( [tempx,tempy]',[tempv]','RBFFunction', 'thinplate');
        rbfcheck_maxdiff = rbfcheck(op2_y);
        if rbfcheck_maxdiff > 1e-3, disp('Please check rbf interpolation! Pause here.'); pause; end
        disp_y = rbfinterp([coordCurr(:,1),coordCurr(:,2)]', op2_y );

        coordCurr = coordCurr + [disp_x(:), disp_y(:)];
        U_accum = (coordCurr - coord)'; U_accum = U_accum(:);
        ResultDisp{ImgSeqNum-1}.U_accum = U_accum; % Store cumulative displacement field

    end
    close(hbar);

else % 'accumulative' mode: ResultDisp{}.U is already cumulative
    for ImgSeqNum = 2 : length(ImgNormalized)
        ResultDisp{ImgSeqNum-1}.U_accum = ResultDisp{ImgSeqNum-1}.U;
    end
end


%% Section 8: Compute strains
fprintf('------------ Section 8 Start ------------ \n')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to compute strain fields and plot disp and strain results
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All parameters below are set in dicpara_default.m and can be overridden
% before running this section. No interactive prompts needed.
% ------ Strain computation ------
Rad = [];
if DICpara.MethodToComputeStrain == 2 % Plane fitting: use half-window size
    Rad = DICpara.StrainPlaneFitRad;
end
% ------ Smoothing ------
if DICpara.smoothness > 0
    DICpara.DoYouWantToSmoothOnceMore = 0;
else
    DICpara.DoYouWantToSmoothOnceMore = 1;
end

%% ====== Start main part ======
% This section is to calculate strain fields based on the transformed
% cumulative displacements: [F] = [D][U]
% [F] = [..., F11_nodei, F21_nodei, F12_nodei, F22_nodei, ...]';
% [u] = [..., U1_nodei, U2_nodei, ...]';
% [D]: finite difference/finite element operator to compute first derivatives

for ImgSeqNum = 2 : length(ImgNormalized)

    close all; disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]);

    %%%%% Load deformed image %%%%%%%
    gNormalizedMask = double( ImgMask{ImgSeqNum} ); % Load the mask file of current deformed frame
    gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask ; % Load current deformed frame
    Dg = img_gradient(gNormalized,gNormalized,gNormalizedMask); % Finite difference to compute image grayscale gradients;

    fNormalizedMask = double( ImgMask{1} ); %
    DICpara.ImgRefMask = fNormalizedMask;

    USubpb2 = ResultDisp{ImgSeqNum-1}.U_accum;
    coordinatesFEM = ResultFEMeshEachFrame{1}.coordinatesFEM;
    elementsFEM = ResultFEMeshEachFrame{1}.elementsFEM;

    if isfield(ResultFEMeshEachFrame{ImgSeqNum-1}, 'markCoordHoleEdge')
        markCoordHoleEdge = ResultFEMeshEachFrame{ImgSeqNum-1}.markCoordHoleEdge;
    end
    DICmesh.coordinatesFEM = coordinatesFEM;
    DICmesh.elementsFEM = elementsFEM;
    coordinatesFEMWorld = DICpara.um2px*[coordinatesFEM(:,1),size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % ------ Plotting and Compute Strain-------
    if size(USubpb2,1) == 1
        ULocal = USubpb2_New.USubpb2; %FLocal = FSubpb2.FSubpb2;
    else
        ULocal = USubpb2;%FLocal = FSubpb2;
    end
    UWorld = DICpara.um2px*ULocal; UWorld(2:2:end) = -UWorld(2:2:end);

    % ------ Smooth displacements ------
    SmoothTimes = 0;

    while DICpara.DoYouWantToSmoothOnceMore == 0 && SmoothTimes < 3
        ULocal = smooth_disp_rbf(ULocal,DICmesh,DICpara);
        SmoothTimes = SmoothTimes + 1;
    end

    % ----- Compute strain field ------
    [FStraintemp, FStrainWorld] = compute_strain(ULocal, [], coordinatesFEM, DICmesh, DICpara, Df, Dg, Rad);

    % ------ Plot disp and strain ------
    if DICpara.OrigDICImgTransparency == 1
        plot_disp_show(UWorld,coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4),DICpara,'NoEdgeColor');
        [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min,strain_maxshear,strain_vonMises] = ...
            plot_strain_no_img(FStrainWorld,coordinatesFEMWorld,elementsFEM(:,1:4),DICpara);

    else % Plot over raw DIC images
        if DICpara.Image2PlotResults == 0 % Plot over the first image; "file_name{1,1}" corresponds to the first image
            plot_disp(UWorld,coordinatesFEMWorld,elementsFEM(:,1:4),file_name{1,1},DICpara);
            [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
                strain_maxshear,strain_vonMises] = plot_strain(UWorld,FStrainWorld, ...
                coordinatesFEMWorld,elementsFEM(:,1:4),file_name{1,1},DICpara);

        else % Plot over second or next deformed images

            fullFilePath = fullfile(file_name{2, ImgSeqNum}, file_name{1, ImgSeqNum});

            %%%%%% New codes: applying mask files %%%%%%
            plot_disp_masks(UWorld,coordinatesFEMWorld,elementsFEM(:,1:4),...
                fullFilePath, ImgMask{ ImgSeqNum },DICpara);

            [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
                strain_maxshear,strain_vonMises] = plot_strain_masks(UWorld,FStrainWorld, ...
                coordinatesFEMWorld,elementsFEM(:,1:4),fullFilePath, ...
                ImgMask{ImgSeqNum },DICpara);

        end
    end

    % ----- Save strain results ------
    ResultStrain{ImgSeqNum-1} = struct('strainxCoord',coordinatesFEMWorld(:,1),'strainyCoord',coordinatesFEMWorld(:,2), ...
        'dispu',UWorld(1:2:end),'dispv',UWorld(2:2:end), ...
        'dudx',FStraintemp(1:4:end),'dvdx',FStraintemp(2:4:end),'dudy',FStraintemp(3:4:end),'dvdy',FStraintemp(4:4:end), ...
        'strain_exx',strain_exx,'strain_exy',strain_exy,'strain_eyy',strain_eyy, ...
        'strain_principal_max',strain_principal_max,'strain_principal_min',strain_principal_min, ...
        'strain_maxshear',strain_maxshear,'strain_vonMises',strain_vonMises);

    % ------ Save figures for tracked displacement and strain fields ------
    DICpara = save_fig_disp_strain(file_name, ImgSeqNum, DICpara);


end
% ------ END of for-loop {ImgSeqNum = 2:length(ImgNormalized)} ------
fprintf('------------ Section 8 Done ------------ \n\n')


% ------ Save data again including solved strain fields ------
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame',...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrain');


%% Section 9: Compute stress
fprintf('------------ Section 9 Start ------------ \n')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to compute stress fields and plot stress fields
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Material model and parameters are set in dicpara_default.m
if (DICpara.MaterialModel == 1) || (DICpara.MaterialModel == 2)
    if isempty(DICpara.MaterialModelPara.YoungsModulus) || isempty(DICpara.MaterialModelPara.PoissonsRatio)
        warning('MaterialModelPara.YoungsModulus and PoissonsRatio must be set in DICpara before Section 9.');
    end
end

% ------ Start main part ------
for ImgSeqNum = 2 : length(ImgNormalized)

    disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]); close all;

    coordinatesFEM = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM;
    elementsFEM = ResultFEMeshEachFrame{ImgSeqNum-1}.elementsFEM;
    coordinatesFEMWorldDef = DICpara.um2px*[coordinatesFEM(:,1),size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)] + ...
        DICpara.Image2PlotResults*[ResultStrain{ImgSeqNum-1}.dispu, ResultStrain{ImgSeqNum-1}.dispv];

    % ------ Plot stress ------
    if DICpara.OrigDICImgTransparency == 1
        [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
            stress_principal_min_xyplane, stress_maxshear_xyplane, ...
            stress_maxshear_xyz3d, stress_vonMises]  =  plot_stress_no_img( ...
            DICpara,ResultStrain{ImgSeqNum-1},coordinatesFEMWorldDef,elementsFEM(:,1:4));

    else % Plot over raw DIC images
        if DICpara.Image2PlotResults == 0 % Plot over the first image; "file_name{1,1}" corresponds to the first image
            [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
                stress_principal_min_xyplane, stress_maxshear_xyplane, ...
                stress_maxshear_xyz3d, stress_vonMises] = plot_stress( ...
                DICpara,ResultStrain{ImgSeqNum-1},coordinatesFEMWorldDef,elementsFEM(:,1:4),file_name{1,1});

        else % Plot over second or next deformed images
            [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
                stress_principal_min_xyplane, stress_maxshear_xyplane, ...
                stress_maxshear_xyz3d, stress_vonMises] = plot_stress( ...
                DICpara,ResultStrain{ImgSeqNum-1},coordinatesFEMWorldDef,elementsFEM(:,1:4),file_name{1,ImgSeqNum});

        end
    end


    % ------ Save figures for computed stress fields ------
    save_fig_stress(file_name, ImgSeqNum, DICpara);

    % ----- Save strain results ------
    ResultStress{ImgSeqNum-1} = struct('stressxCoord',ResultStrain{ImgSeqNum-1}.strainxCoord,'stressyCoord',ResultStrain{ImgSeqNum-1}.strainyCoord, ...
        'stress_sxx',stress_sxx,'stress_sxy',stress_sxy,'stress_syy',stress_syy, ...
        'stress_principal_max_xyplane',stress_principal_max_xyplane, 'stress_principal_min_xyplane',stress_principal_min_xyplane, ...
        'stress_maxshear_xyplane',stress_maxshear_xyplane,'stress_maxshear_xyz3d',stress_maxshear_xyz3d, ...
        'stress_vonMises',stress_vonMises);

end
% ------ END of for-loop {ImgSeqNum = 2:length(ImgNormalized)} ------
fprintf('------------ Section 9 Done ------------ \n\n')

% ------ Save data again including solved stress fields ------
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame', ...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrain','ResultStress');


%% Section 10: Plot the generated quadtree mesh
figure,
for ImgSeqNum = 2 : (1+size(ResultDisp,1))

    clf; patch('Faces', DICmesh.elementsFEM(:,1:4), 'Vertices', DICmesh.coordinatesFEMWorld + ...
        [ResultDisp{ImgSeqNum-1}.U(1:2:end), -ResultDisp{ImgSeqNum-1}.U(2:2:end)], 'Facecolor','none','linewidth',1)
    xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
    tt = title(['Frame #',num2str(ImgSeqNum)],'fontweight','normal');
    set(tt,'Interpreter','latex','fontsize',10);
    axis equal; axis tight; set(gca,'fontsize',18); set(gcf,'color','w'); box on;
    a = gca; a.TickLabelInterpreter = 'latex';

