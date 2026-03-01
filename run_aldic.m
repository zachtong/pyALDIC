function results = run_aldic(DICpara, file_name, Img, ImgMask, varargin)
%run_aldic  Execute the full AL-DIC pipeline.
%
%   results = run_aldic(DICpara, file_name, Img, ImgMask)
%   results = run_aldic(..., 'ProgressFcn', @(frac,msg) ...)
%   results = run_aldic(..., 'StopFcn', @() false)
%
%   INPUTS:
%       DICpara   - struct from dicpara_default (with all fields set)
%       file_name - cell array of image file names {6 x nFrames} from read_images
%       Img       - cell array of loaded images (double, transposed)
%       ImgMask   - cell array of mask images (logical, transposed)
%
%   Name-value pairs:
%     'ProgressFcn'    function handle(fraction, message) for progress updates
%     'StopFcn'        function handle() returning true to abort computation
%     'ComputeStrain'  logical (default true). If false, skip strain computation
%                      and return results without ResultStrain field.
%
%   OUTPUT:
%       results - struct with fields:
%         .DICpara, .DICmesh, .file_name
%         .ResultDisp, .ResultDefGrad, .ResultStrain
%         .ResultFEMesh, .ResultFEMeshEachFrame
%
% ----------------------------------------------
% Original author: Jin Yang, PhD @Caltech
% Refactored: 02/2026
% ==============================================

    %% Parse name-value pairs
    p = inputParser;
    addParameter(p, 'ProgressFcn', @(frac,msg) default_progress(frac,msg));
    addParameter(p, 'StopFcn', @() false);
    addParameter(p, 'ComputeStrain', true);
    parse(p, varargin{:});
    progressFcn = p.Results.ProgressFcn;
    stopFcn = p.Results.StopFcn;
    computeStrain = p.Results.ComputeStrain;
    showPlots = DICpara.showPlots;

    %% Section 2b: Normalize images and initialize storage
    fprintf('------------ Section 2 Start ------------ \n')
    [ImgNormalized, DICpara.gridxyROIRange] = normalize_img(Img, DICpara.gridxyROIRange);

    % Auto-scale FFT search region if too large for image size
    imgMinDim = min(DICpara.ImgSize);
    maxSafeSearch = floor(imgMinDim/4 - max(DICpara.winsize));
    if DICpara.SizeOfFFTSearchRegion > maxSafeSearch
        oldVal = DICpara.SizeOfFFTSearchRegion;
        DICpara.SizeOfFFTSearchRegion = max(10, maxSafeSearch);
        warnMsg = sprintf('Auto-scaled FFT search region: %d -> %d (image size [%d x %d])', ...
            oldVal, DICpara.SizeOfFFTSearchRegion, DICpara.ImgSize(1), DICpara.ImgSize(2));
        warning('run_aldic:fftAutoScale', '%s', warnMsg);
        progressFcn(0, warnMsg);
    end

    ResultDisp = cell(length(ImgNormalized)-1, 1);
    ResultDefGrad = cell(length(ImgNormalized)-1, 1);
    ResultStrain = cell(length(ImgNormalized)-1, 1);
    ResultFEMeshEachFrame = cell(length(ImgNormalized)-1, 1);
    ResultFEMesh = cell(ceil((length(ImgNormalized)-1)/DICpara.ImgSeqIncUnit), 1);
    fprintf('------------ Section 2 Done ------------ \n\n')

    %% Debug options
    UseGlobal = DICpara.UseGlobalStep;

    %% ========================================================================
    % Main frame loop (Sections 3-6)
    % =========================================================================
    nFrames = length(ImgNormalized);
    for ImgSeqNum = 2 : nFrames

        % --- Stop check ---
        if stopFcn()
            fprintf('Computation stopped by user at frame %d.\n', ImgSeqNum);
            break;
        end

        if showPlots, close all; end
        disp(['Current image frame #: ', num2str(ImgSeqNum), '/', num2str(nFrames)]);
        progressFcn((ImgSeqNum-2)/(nFrames-1), sprintf('Processing frame %d/%d', ImgSeqNum, nFrames));

        % ====== Load reference image ======
        if strcmp(DICpara.referenceMode, 'accumulative')
            fNormalizedMask = double(ImgMask{1});
            fNormalized = ImgNormalized{1} .* fNormalizedMask;
        else % 'incremental'
            fNormalizedMask = double(ImgMask{ImgSeqNum-1});
            fNormalized = ImgNormalized{ImgSeqNum-1} .* fNormalizedMask;
        end
        Df = img_gradient(fNormalized, fNormalized, fNormalizedMask);

        gNormalizedMask = double(ImgMask{ImgSeqNum});
        gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask;
        DICpara.ImgRefMask = fNormalizedMask;

        if showPlots
            figure,
            subplot(2,2,1); imshow(fNormalized'); title('fNormalized'); colorbar;
            subplot(2,2,2); imshow(gNormalized'); title('gNormalized'); colorbar;
            subplot(2,2,3); imshow(fNormalizedMask'); title('f mask'); colorbar;
            subplot(2,2,4); imshow(gNormalizedMask'); title('g mask'); colorbar;
        end


        %% Section 3: Compute initial guess
        fprintf('\n'); fprintf('------------ Section 3 Start ------------ \n')

        if ImgSeqNum == 2 || DICpara.NewFFTSearch == 1
            DICpara.InitFFTSearchMethod = 1;
            [DICpara,x0temp_f,y0temp_f,u_f,v_f,cc] = integer_search(fNormalized,gNormalized,file_name,DICpara);

            % =================== Zach improvement starts 20250624 ================
            xnodes = max([1+0.5*DICpara.winsize,DICpara.gridxyROIRange.gridx(1)]) ...
                : DICpara.winstepsize : min([size(fNormalized,1)-0.5*DICpara.winsize-1,DICpara.gridxyROIRange.gridx(2)]);
            ynodes = max([1+0.5*DICpara.winsize,DICpara.gridxyROIRange.gridy(1)]) ...
                : DICpara.winstepsize : min([size(fNormalized,2)-0.5*DICpara.winsize-1,DICpara.gridxyROIRange.gridy(2)]);

            [x0temp,y0temp] = ndgrid(xnodes,ynodes);

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
            % =================== Zach improvement ends ===================

            [DICmesh] = mesh_setup(x0temp, y0temp, DICpara);
            U0 = init_disp(u, v, cc.max, DICmesh.x0, DICmesh.y0, 0);

            % Set zero at holes
            linearIndices1 = sub2ind(size(fNormalizedMask), DICmesh.coordinatesFEM(:,1), DICmesh.coordinatesFEM(:,2));
            MaskOrNot1 = fNormalizedMask(linearIndices1);
            nanIndex = find(MaskOrNot1 < 1);
            U0(2*nanIndex) = nan;
            U0(2*nanIndex-1) = nan;

            % Deal with incremental mode
            fNormalizedNewIndex = ImgSeqNum - mod(ImgSeqNum-2, DICpara.ImgSeqIncUnit) - 1;
            if DICpara.ImgSeqIncUnit == 1, fNormalizedNewIndex = fNormalizedNewIndex - 1; end
            ResultFEMesh{1+floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)} = ...
                struct('coordinatesFEM', DICmesh.coordinatesFEM, 'elementsFEM', DICmesh.elementsFEM, ...
                'winsize', DICpara.winsize, 'winstepsize', DICpara.winstepsize, 'gridxyROIRange', DICpara.gridxyROIRange);

            % Generate quadtree mesh
            DICmesh.elementMinSize = DICpara.winsizeMin;
            [DICmesh, U0] = generate_mesh(DICmesh, DICpara, Df, U0);
            if showPlots
                plot_disp_show(U0, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
            end

            ResultFEMeshEachFrame{ImgSeqNum-1} = struct('coordinatesFEM', DICmesh.coordinatesFEM, ...
                'elementsFEM', DICmesh.elementsFEM, 'markCoordHoleEdge', DICmesh.markCoordHoleEdge);

        else
            % Reuse previous mesh, predict U0
            if DICpara.usePODGPR && ImgSeqNum >= DICpara.POD_startFrame
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
                [u_pred,~,~,~] = por_gpr(T_data_u, t_train, t_pre, nB);
                [v_pred,~,~,~] = por_gpr(T_data_v, t_train, t_pre, nB);
                tempu = u_pred(1,:); tempv = v_pred(1,:);
                U0 = [tempu(:),tempv(:)]'; U0 = U0(:);
                disp('POD-GPR prediction used for initial guess.');
            else
                U0 = ResultDisp{ImgSeqNum-2}.U;
                disp('Previous frame result used as initial guess.');
            end
            ResultFEMeshEachFrame{ImgSeqNum-1} = struct('coordinatesFEM', DICmesh.coordinatesFEM, ...
                'elementsFEM', DICmesh.elementsFEM, 'markCoordHoleEdge', DICmesh.markCoordHoleEdge);
        end

        fprintf('------------ Section 3 Done ------------ \n\n')


        %% Section 4: ALDIC Subproblem 1
        fprintf('------------ Section 4 Start ------------ \n')
        mu=0; beta=0; tol=DICpara.tol; ALSolveStep=1; ALSub1Time=zeros(6,1); ALSub2Time=zeros(6,1);
        ConvItPerEle=zeros(size(DICmesh.coordinatesFEM,1),6); ALSub1BadPtNum=zeros(6,1);
        disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****'])

        [USubpb1,FSubpb1,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp,markCoordHoleStrain] = ...
            local_icgn(U0,DICmesh.coordinatesFEM,Df,fNormalized,gNormalized,DICpara,tol);
        ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp; ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp; toc

        coordinatesFEM = DICmesh.coordinatesFEM;
        U = USubpb1; F = FSubpb1;

        USubpb1World = USubpb1; USubpb1World(2:2:end) = -USubpb1(2:2:end); FSubpb1World = FSubpb1;

        if showPlots
            plot_disp_show(USubpb1World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
            plot_strain_show(FSubpb1World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
        end

        % Store step data in memory (no temp .mat files)
        stepData = struct();
        stepData(ALSolveStep).USubpb1 = USubpb1;
        stepData(ALSolveStep).FSubpb1 = FSubpb1;

        fprintf('------------ Section 4 Done ------------ \n\n')


        if UseGlobal
            %% Section 5: Subproblem 2
            fprintf('------------ Section 5 Start ------------ \n'); tic;

            LevelNo=1;
            if DICpara.DispSmoothness>1e-6, USubpb1 = smooth_disp_rbf(USubpb1,DICmesh,DICpara); end
            if DICpara.StrainSmoothness>1e-6, FSubpb1 = smooth_strain_rbf(FSubpb1,DICmesh,DICpara); end

            mu = DICpara.mu; udual = 0*FSubpb1; vdual = 0*USubpb1;
            betaList = DICpara.betaRange * mean(DICpara.winstepsize).^2 .* mu;
            Err1 = zeros(length(betaList),1); Err2 = Err1;

            disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****']);
            alpha = DICpara.alpha;

            if ImgSeqNum == 2
                for tempk = 1:length(betaList)
                    beta = betaList(tempk); display(['Try #',num2str(tempk),' beta = ',num2str(beta)]);
                    alpha=0; [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize));
                    FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);
                    Err1(tempk) = norm(USubpb1-USubpb2,2);
                    Err2(tempk) = norm(FSubpb1-FSubpb2,2);
                end

                Err1Norm = (Err1-mean(Err1))/std(Err1);
                Err2Norm = (Err2-mean(Err2))/std(Err2);
                ErrSum = Err1Norm+Err2Norm;
                [~,indexOfbeta] = min(ErrSum);

                try
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

            if abs(beta-betaList(end))>abs(eps)
                [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize));
                FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);
                ALSub2Time(ALSolveStep) = toc; toc
            end

            if DICpara.DispSmoothness>1e-6, USubpb2 = smooth_disp_rbf(USubpb2,DICmesh,DICpara); end
            for tempk=0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
            if DICpara.StrainSmoothness>1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1,DICmesh,DICpara); end
            for tempk=0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
            for tempk=0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end

            % Store step data in memory
            stepData(ALSolveStep).USubpb2 = USubpb2;
            stepData(ALSolveStep).FSubpb2 = FSubpb2;
            udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
            stepData(ALSolveStep).udual = udual;
            stepData(ALSolveStep).vdual = vdual;
            fprintf('------------ Section 5 Done ------------ \n\n')


            %% Section 6: ADMM iterations
            fprintf('------------ Section 6 Start ------------ \n')
            ALSolveStep = 1; tol2 = DICpara.ADMM_tol; UpdateY = 1e4;

            while (ALSolveStep < DICpara.ADMM_maxIter)
                ALSolveStep = ALSolveStep + 1;

                winsize_List = DICpara.winsize*ones(size(DICmesh.coordinatesFEM,1),2);
                DICpara.winsize_List = winsize_List;

                % Subproblem 1
                disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****']);
                tic; [USubpb1,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp] = subpb1_solver(...
                    USubpb2,FSubpb2,udual,vdual,DICmesh.coordinatesFEM,...
                    Df,fNormalized,gNormalized,mu,beta,DICpara,tol);
                FSubpb1 = FSubpb2; toc
                ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp; ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp;
                stepData(ALSolveStep).USubpb1 = USubpb1;
                stepData(ALSolveStep).FSubpb1 = FSubpb1;

                % Subproblem 2
                disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****'])
                tic; [USubpb2] = subpb2_solver(DICmesh,DICpara.GaussPtOrder,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,mean(DICpara.winstepsize));
                FSubpb2 = global_nodal_strain_rbf(DICmesh,DICpara,USubpb2);
                ALSub2Time(ALSolveStep) = toc; toc

                if DICpara.DispSmoothness>1e-6, USubpb2 = smooth_disp_rbf(USubpb2,DICmesh,DICpara); end
                for tempk=0:3, FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk); end
                if DICpara.StrainSmoothness>1e-6, FSubpb2 = smooth_strain_rbf(0.1*FSubpb2+0.9*FSubpb1,DICmesh,DICpara); end
                for tempk=0:1, USubpb2(2*markCoordHoleStrain-tempk) = USubpb1(2*markCoordHoleStrain-tempk); end
                for tempk=0:3, FSubpb2(4*markCoordHoleStrain-tempk) = FSubpb1(4*markCoordHoleStrain-tempk); end

                stepData(ALSolveStep).USubpb2 = USubpb2;
                stepData(ALSolveStep).FSubpb2 = FSubpb2;

                % Convergence check (in-memory, no file I/O)
                if (mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) ~= 0 && (ImgSeqNum>2)) || (ImgSeqNum < DICpara.ImgSeqIncUnit)
                    UpdateY = norm(stepData(ALSolveStep-1).USubpb2 - stepData(ALSolveStep).USubpb2, 2) / sqrt(length(stepData(ALSolveStep).USubpb2));
                    UpdateY2 = norm(stepData(ALSolveStep-1).USubpb1 - stepData(ALSolveStep).USubpb1, 2) / sqrt(length(stepData(ALSolveStep).USubpb1));
                end
                if exist('UpdateY','var'), disp(['Update global step = ',num2str(UpdateY)]); end
                if exist('UpdateY2','var'), disp(['Update local step  = ',num2str(UpdateY2)]); end
                fprintf('*********************************** \n\n');

                udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
                stepData(ALSolveStep).udual = udual;
                stepData(ALSolveStep).vdual = vdual;

                if exist('UpdateY','var') && exist('UpdateY2','var')
                    if UpdateY < tol2 || UpdateY2 < tol2
                        break
                    end
                end
            end
            fprintf('------------ Section 6 Done ------------ \n\n')
        end

        % Save frame results
        if UseGlobal
            ResultDisp{ImgSeqNum-1}.U = full(USubpb2);
            ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
            ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb2);
        else
            ResultDisp{ImgSeqNum-1}.U = full(USubpb1);
            ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
            ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb1);
        end

    end
    % ------ End of main frame loop ------


    %% Plot last frame results
    if showPlots
        USubpb2World = USubpb2; USubpb2World(2:2:end) = -USubpb2(2:2:end); FSubpb2World = FSubpb2;
        close all; plot_disp_show(USubpb2World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
        plot_strain_show(FSubpb2World, DICmesh.coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'EdgeColor');
    end

    progressFcn(0.5, 'Frame loop done. Checking convergence...');


    %% Section 7: Check convergence (in-memory)
    fprintf('------------ Section 7 Start ------------ \n')
    fprintf('***** Check convergence ***** \n');
    ALSolveStep1 = min(6, ALSolveStep);

    disp('==== uhat^(k) - u^(k) ====');
    for stepk = 1:ALSolveStep1
        UpdateY_s = norm(stepData(stepk).USubpb2 - stepData(stepk).USubpb1, 2) / sqrt(length(stepData(stepk).USubpb2));
        disp(num2str(UpdateY_s));
    end
    disp('==== Fhat^(k) - F^(k) ====');
    for stepk = 1:ALSolveStep1
        UpdateF_s = norm(stepData(stepk).FSubpb1 - stepData(stepk).FSubpb2, 2) / sqrt(length(stepData(stepk).FSubpb1));
        disp(num2str(UpdateF_s));
    end
    disp('==== uhat^(k) - uhat^(k-1) ====');
    for stepk = 2:ALSolveStep1
        UpdateY_s = norm(stepData(stepk-1).USubpb2 - stepData(stepk).USubpb2, 2) / sqrt(length(stepData(stepk).USubpb2));
        disp(num2str(UpdateY_s));
    end
    disp('==== udual^(k) - udual^(k-1) ====');
    for stepk = 2:ALSolveStep1
        UpdateW_s = norm(stepData(stepk-1).udual - stepData(stepk).udual, 2) / sqrt(length(stepData(stepk).udual));
        disp(num2str(UpdateW_s));
    end
    disp('==== vdual^(k) - vdual^(k-1) ====');
    for stepk = 2:ALSolveStep1
        Updatev_s = norm(stepData(stepk-1).vdual - stepData(stepk).vdual, 2) / sqrt(length(stepData(stepk).vdual));
        disp(num2str(Updatev_s));
    end
    fprintf('------------ Section 7 Done ------------ \n\n')

    clear a ALSub1BadPtNum ALSub1Timetemp atemp b btemp cc ConvItPerEletemp
    clear coordinatesFEMQuadtree elementsFEMQuadtree


    %% Transform displacement fields to cumulative
    progressFcn(0.6, 'Computing cumulative displacements...');
    if strcmp(DICpara.referenceMode, 'incremental')
        tempx = ResultFEMeshEachFrame{1}.coordinatesFEM(:,1);
        tempy = ResultFEMeshEachFrame{1}.coordinatesFEM(:,2);
        coord = [tempx,tempy]; coordCurr = coord;
        if showPlots
            hbar = waitbar(0, 'Calculate cumulative disp from incremental disp');
        end

        for ImgSeqNum = 2 : nFrames
            if showPlots, waitbar((ImgSeqNum-1)/(nFrames-1)); end
            tempx = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM(:,1);
            tempy = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM(:,2);
            tempu = ResultDisp{ImgSeqNum-1}.U(1:2:end);
            tempv = ResultDisp{ImgSeqNum-1}.U(2:2:end);

            op2_x = rbfcreate([tempx,tempy]', [tempu]', 'RBFFunction', 'thinplate');
            rbfcheck_maxdiff = rbfcheck(op2_x);
            if rbfcheck_maxdiff > 1e-3, warning('run_aldic:rbfCheck', 'RBF interpolation maxdiff=%.4f > 1e-3.', rbfcheck_maxdiff); end
            disp_x = rbfinterp([coordCurr(:,1),coordCurr(:,2)]', op2_x);

            op2_y = rbfcreate([tempx,tempy]', [tempv]', 'RBFFunction', 'thinplate');
            rbfcheck_maxdiff = rbfcheck(op2_y);
            if rbfcheck_maxdiff > 1e-3, warning('run_aldic:rbfCheck', 'RBF interpolation maxdiff=%.4f > 1e-3.', rbfcheck_maxdiff); end
            disp_y = rbfinterp([coordCurr(:,1),coordCurr(:,2)]', op2_y);

            coordCurr = coordCurr + [disp_x(:), disp_y(:)];
            U_accum = (coordCurr - coord)'; U_accum = U_accum(:);
            ResultDisp{ImgSeqNum-1}.U_accum = U_accum;
        end
        if showPlots, close(hbar); end
    else
        for ImgSeqNum = 2 : nFrames
            ResultDisp{ImgSeqNum-1}.U_accum = ResultDisp{ImgSeqNum-1}.U;
        end
    end


    %% Section 8: Compute world-space displacements (always) and strains (optional)
    fprintf('------------ Section 8 Start ------------ \n')

    coordinatesFEM = ResultFEMeshEachFrame{1}.coordinatesFEM;
    elementsFEM = ResultFEMeshEachFrame{1}.elementsFEM;
    DICmesh.coordinatesFEM = coordinatesFEM;
    DICmesh.elementsFEM = elementsFEM;
    coordinatesFEMWorld = DICpara.um2px*[coordinatesFEM(:,1), size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)];

    if computeStrain
        progressFcn(0.7, 'Computing strains...');
        Rad = [];
        if DICpara.MethodToComputeStrain == 2
            Rad = DICpara.StrainPlaneFitRad;
        end
        if DICpara.smoothness > 0
            DICpara.DoYouWantToSmoothOnceMore = 0;
        else
            DICpara.DoYouWantToSmoothOnceMore = 1;
        end
    else
        progressFcn(0.7, 'Computing world-space displacements (strain skipped)...');
    end

    for ImgSeqNum = 2 : nFrames
        if showPlots, close all; end
        disp(['Current image frame #: ', num2str(ImgSeqNum), '/', num2str(nFrames)]);

        USubpb2 = ResultDisp{ImgSeqNum-1}.U_accum;
        if size(USubpb2,1) == 1
            ULocal = USubpb2_New.USubpb2;
        else
            ULocal = USubpb2;
        end
        UWorld = DICpara.um2px*ULocal; UWorld(2:2:end) = -UWorld(2:2:end);

        if computeStrain
            % Compute image gradients for strain
            gNormalizedMask = double(ImgMask{ImgSeqNum});
            gNormalized = ImgNormalized{ImgSeqNum} .* gNormalizedMask;
            Dg = img_gradient(gNormalized, gNormalized, gNormalizedMask);
            fNormalizedMask = double(ImgMask{1});
            DICpara.ImgRefMask = fNormalizedMask;

            % Smooth displacements
            SmoothTimes = 0;
            while DICpara.DoYouWantToSmoothOnceMore == 0 && SmoothTimes < 3
                ULocal = smooth_disp_rbf(ULocal, DICmesh, DICpara);
                SmoothTimes = SmoothTimes + 1;
            end

            % Compute strain field
            [FStraintemp, FStrainWorld] = compute_strain(ULocal, [], coordinatesFEM, DICmesh, DICpara, Df, Dg, Rad);

            % Compute strain components
            if showPlots
                if DICpara.OrigDICImgTransparency == 1
                    plot_disp_show(UWorld, coordinatesFEMWorld, DICmesh.elementsFEM(:,1:4), DICpara, 'NoEdgeColor');
                    [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min,strain_maxshear,strain_vonMises] = ...
                        plot_strain_no_img(FStrainWorld, coordinatesFEMWorld, elementsFEM(:,1:4), DICpara);
                else
                    if DICpara.Image2PlotResults == 0
                        plot_disp(UWorld, coordinatesFEMWorld, elementsFEM(:,1:4), file_name{1,1}, DICpara);
                        [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
                            strain_maxshear,strain_vonMises] = plot_strain(UWorld, FStrainWorld, ...
                            coordinatesFEMWorld, elementsFEM(:,1:4), file_name{1,1}, DICpara);
                    else
                        fullFilePath = fullfile(file_name{2,ImgSeqNum}, file_name{1,ImgSeqNum});
                        plot_disp_masks(UWorld, coordinatesFEMWorld, elementsFEM(:,1:4), ...
                            fullFilePath, ImgMask{ImgSeqNum}, DICpara);
                        [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
                            strain_maxshear,strain_vonMises] = plot_strain_masks(UWorld, FStrainWorld, ...
                            coordinatesFEMWorld, elementsFEM(:,1:4), fullFilePath, ...
                            ImgMask{ImgSeqNum}, DICpara);
                    end
                end
                DICpara = save_fig_disp_strain(file_name, ImgSeqNum, DICpara);
            else
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
            end

            ResultStrain{ImgSeqNum-1} = struct('strainxCoord',coordinatesFEMWorld(:,1),'strainyCoord',coordinatesFEMWorld(:,2), ...
                'dispu',UWorld(1:2:end),'dispv',UWorld(2:2:end), ...
                'dudx',FStraintemp(1:4:end),'dvdx',FStraintemp(2:4:end),'dudy',FStraintemp(3:4:end),'dvdy',FStraintemp(4:4:end), ...
                'strain_exx',strain_exx,'strain_exy',strain_exy,'strain_eyy',strain_eyy, ...
                'strain_principal_max',strain_principal_max,'strain_principal_min',strain_principal_min, ...
                'strain_maxshear',strain_maxshear,'strain_vonMises',strain_vonMises);
        else
            % Displacement-only: store world-space displacements without strain
            ResultStrain{ImgSeqNum-1} = struct('strainxCoord',coordinatesFEMWorld(:,1),'strainyCoord',coordinatesFEMWorld(:,2), ...
                'dispu',UWorld(1:2:end),'dispv',UWorld(2:2:end));
        end
    end
    fprintf('------------ Section 8 Done ------------ \n\n')


    %% Assemble output struct
    results.DICpara = DICpara;
    results.DICmesh = DICmesh;
    results.file_name = file_name;
    results.ResultDisp = ResultDisp;
    results.ResultDefGrad = ResultDefGrad;
    results.ResultStrain = ResultStrain;
    results.ResultFEMesh = ResultFEMesh;
    results.ResultFEMeshEachFrame = ResultFEMeshEachFrame;
    results.ALSub1Time = ALSub1Time;
    results.ALSub2Time = ALSub2Time;
    results.ALSolveStep = ALSolveStep;

    progressFcn(1.0, 'Pipeline complete.');

end


%% ====== Helper functions ======

function default_progress(frac, msg)
%DEFAULT_PROGRESS  Print progress to console.
    fprintf('[%3.0f%%] %s\n', frac*100, msg);
end
