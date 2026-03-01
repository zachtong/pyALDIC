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

% ------ Clear bad points for Local DIC ------
% find bad points after Local Subset ICGN
if isfield(DICpara, 'ICGNMaxIter')
    maxIterNum = DICpara.ICGNMaxIter;
else
    maxIterNum = 100;
end
[row1,~] = find(ConvItPerEle(:)<0);
[row2,~] = find(ConvItPerEle(:)>maxIterNum-1);
[row3,~] = find(ConvItPerEle(:)==maxIterNum+2);
LocalICGNBadPt = unique(union(row1,row2)); LocalICGNBadPtNum = length(LocalICGNBadPt)-length(row3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Though some subsets are converged, but their accuracy is worse than most
% other subsets. This step is to remove those subsets with abnormal convergence steps
LocalICGNGoodPt = setdiff([1:1:size(coordinatesFEM,1)],LocalICGNBadPt);
ConvItPerEleMean = mean(ConvItPerEle(LocalICGNGoodPt));
ConvItPerEleStd = std(ConvItPerEle(LocalICGNGoodPt));
if isfield(DICpara, 'outlierSigmaFactor'), osf = DICpara.outlierSigmaFactor; else, osf = 0.25; end
if isfield(DICpara, 'outlierMinThreshold'), omt = DICpara.outlierMinThreshold; else, omt = 10; end
[row4,~] = find(ConvItPerEle(:) > max([ConvItPerEleMean + osf*ConvItPerEleStd, omt]));
LocalICGNBadPt = unique(union(LocalICGNBadPt,row4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-length(row3)), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3))),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN;

%%%%%% Fill nans %%%%%%
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
 
% dilatedI = ( imgaussfilt(double(Df.ImgRefMask),0.5) );
% dilatedI = logical( dilatedI > 0.01); % figure, imshow(dilatedI)
dilatedI = logical(DICpara.ImgRefMask);
cc = bwconncomp(dilatedI,8);
indPxAll = sub2ind( Df.imgSize, round(coordinatesFEM(:,1)),round(coordinatesFEM(:,2)));
indPxNotNanAll = sub2ind( Df.imgSize, round(coordinatesFEM(notnanindex,1)), round(coordinatesFEM(notnanindex,2)) );
stats = regionprops(cc,'Area','PixelList'); 

for tempi = 1:length(stats)
    
    try % if stats(tempi).Area > 20
        
        %%%%% Find those nodes %%%%%
        indPxtempi = sub2ind( Df.imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
        Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
        Lib = ismember(indPxNotNanAll,indPxtempi); [LibList,~] = find(Lib==1);

        %%%%% RBF (Radial basis function) works better than "scatteredInterpolant" %%%%%
        % ------ Disp u ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList)-1)]','RBFFunction', 'thinplate'); rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        U(2*LiaList-1) = fi1(:);

        % ------ Disp v ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList) )]','RBFFunction', 'thinplate'); rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        U(2*LiaList ) = fi1(:);

    catch ME
        warning('subpb1_solver:rbfNanFill', 'RBF NaN fill failed for region %d: %s', tempi, ME.message);
    end

end

%% Final NaN filling (scatteredInterpolant fallback)
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);

if ~isempty(nanindex) && ~isempty(notnanindex)

    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'natural','nearest');
    U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'natural','nearest');
    U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

    U = [U1(:),U2(:)]'; U = U(:);
elseif ~isempty(nanindex) && isempty(notnanindex)
    % All nodes are NaN (all ICGN calls failed) — fall back to previous result
    warning('subpb1_solver:allNaN', 'All ICGN nodes failed, falling back to UOld.');
    U = UOld(:);
end





