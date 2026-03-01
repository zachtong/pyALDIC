function [U,ALSub1Time,ConvItPerEle,LocalICGNBadPtNum] = subpb1_solver(UOld,FOld,udual,vdual,coordinatesFEM,...
                                            Df,ImgRef,ImgDef,mu,beta,DICpara,tol)
%FUNCTION [U,ALSub1Time,ConvItPerEle,LocalICGNBadPtNum] = subpb1_solver(UOld,FOld,udual,vdual,coordinatesFEM,...
%                                            Df,ImgRef,ImgDef,mu,beta,DICpara,tol)
% The ALDIC Subproblem 1 ICGN subset solver (part I): to assign a sequential or a parallel computing
% (see part II: ./solver/icgn_subpb1.m)
% ----------------------------------------------
%   INPUT: UOld                 Initial guess of the displacement fields
%          FOld                 Initial guess of the deformation gradients
%          udual,vdual          Dual variables
%          coordinatesFEM       FE mesh coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          mu,beta              ALDIC coefficients
%          DICpara              DIC parameters: subset size, subset spacing, ...
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           ALSub1Time          Computation time
%           ConvItPerEle        ICGN iteration step for convergence
%           LocalICGNBadPtNum   Number of subsets whose ICGN iterations don't converge
%
% ----------------------------------------------
% Reference
% [1] RegularizeNd. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/61436-regularizend
% [2] Gridfit. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
% [3] Rbfinterp. Matlab File Exchange open source.
% https://www.mathworks.com/matlabcentral/fileexchange/10056-scattered-data-interpolation-and-approximation-using-radial-base-functions
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================


%% Initialization
warnState = warning('off', 'MATLAB:nearlySingularMatrix');
cleanupWarn = onCleanup(@() warning(warnState));
winsize = DICpara.winsize;
winstepsize = DICpara.winstepsize;
ClusterNo = DICpara.ClusterNo;
if isfield(DICpara, 'showPlots'), showWaitbar = DICpara.showPlots; else, showWaitbar = true; end

temp = zeros(size(coordinatesFEM,1),1); UPar = cell(2,1); UPar{1} = temp; UPar{2} = temp;
ConvItPerEle = zeros(size(coordinatesFEM,1),1);
 
% Update winsize for each subset
winsize_x = DICpara.winsize_List(:,1);
winsize_y = DICpara.winsize_List(:,2);

% %%%%% Some old codes dealing with parallel pools %%%%%
% disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****'])
% ------ Within each iteration step ------
% disp('This step takes not short time, please drink coffee and wait.'); tic;
% -------- How to change parallel pools ---------
% myCluster = parcluster('local');
% myCluster.NumWorkers = 4;  % 'Modified' property now TRUE
% saveProfile(myCluster);    % 'local' profile now updated,
%                            % 'Modified' property now FALSE
% -------------- Or we can do this --------------
% Go to the Parallel menu, then select Manage Cluster Profiles.
% Select the "local" profile, and change NumWorkers to 4.
% -----------------------------------------------

%% ClusterNo == 0 or 1: Sequential computing
if (ClusterNo == 0) || (ClusterNo == 1)

    if showWaitbar, h = waitbar(0,'Please wait for Subproblem 1 IC-GN iterations!'); end
    tic;

    for tempj = 1 : size(coordinatesFEM,1)  % tempj is the element index
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2));

        try
            [Utemp,ConvItPerEle(tempj,:)] = icgn_subpb1( ...
                x0temp,y0temp,Df,ImgRef,ImgDef,winsize_x(tempj),winsize_y(tempj),...
                beta,mu,udual(4*tempj-3:4*tempj),vdual(2*tempj-1:2*tempj),...
                UOld(2*tempj-1:2*tempj),FOld(4*tempj-3:4*tempj),tol);

            UPar{1}(tempj) = Utemp(1); UPar{2}(tempj) = Utemp(2);
        catch
            UPar{1}(tempj) = nan; UPar{2}(tempj) = nan;
        end
    end
    if showWaitbar, close(h); end
    ALSub1Time = toc;
    

%% ClusterNo > 1: parallel computing
else

    if showWaitbar, hbar = parfor_progressbar(size(coordinatesFEM,1),'Please wait for Subproblem 1 IC-GN iterations!'); end
    UtempPar = UPar{1}; VtempPar = UPar{2};

    parfor tempj = 1:size(coordinatesFEM,1)
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2));

        try
            [Utemp,ConvItPerEle(tempj,:)] = icgn_subpb1( ...
                x0temp,y0temp,Df,ImgRef,ImgDef,winsize_x(tempj),winsize_y(tempj),...
                beta,mu,udual(4*tempj-3:4*tempj),vdual(2*tempj-1:2*tempj),...
                UOld(2*tempj-1:2*tempj),FOld(4*tempj-3:4*tempj),tol);

            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2);
        catch
            UtempPar(tempj) = nan; VtempPar(tempj) = nan;
        end
        if showWaitbar, hbar.iterate(1); end
    end
    if showWaitbar, close(hbar); end
    ALSub1Time = toc;
    UPar{1} = UtempPar; UPar{2} = VtempPar;

end

U = UOld(:);
U(1:2:end) = UPar{1}; U(2:2:end) = UPar{2};

% ------ Clear bad points ------
if isfield(DICpara, 'ICGNMaxIter'), maxIterNum = DICpara.ICGNMaxIter; else, maxIterNum = 100; end
if isfield(DICpara, 'outlierSigmaFactor'), osf = DICpara.outlierSigmaFactor; else, osf = 0.25; end
if isfield(DICpara, 'outlierMinThreshold'), omt = DICpara.outlierMinThreshold; else, omt = 10; end
[LocalICGNBadPt, LocalICGNBadPtNum] = detect_bad_points(ConvItPerEle, maxIterNum, coordinatesFEM, osf, omt);

nMaskOnly = length(find(ConvItPerEle(:)==maxIterNum+2));
disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-nMaskOnly), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-nMaskOnly)),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN;

% Fill NaN displacements
U = fill_nan_rbf(U, coordinatesFEM, Df.imgSize, DICpara.ImgRefMask, 2);





