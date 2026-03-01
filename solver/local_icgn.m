function [U,F,LocalTime,ConvItPerEle,LocalICGNBadPtNum,markCoordHoleStrain] = ...
                    local_icgn(U0,coordinatesFEM,Df,ImgRef,ImgDef,DICpara,tol)
%FUNCTION [U,F,LocalTime,ConvItPerEle,LocalICGNBadPtNum,markCoordHoleStrain] = ...
%              local_icgn(U0,coordinatesFEM,Df,ImgRef,ImgDef,DICpara,tol)
% The Local ICGN subset solver: dispatches sequential or parallel computing
% (see solver: ./solver/icgn_solver.m)
% ----------------------------------------------
%   INPUT: U0                   Initial guess of the displacement fields
%          coordinatesFEM       FE mesh coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          DICpara              DIC parameters: subset size, subset spacing, ...
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           F                   Deformation gradient tensor
%                               F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%           LocalTime           Computation time
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

temp = zeros(size(coordinatesFEM,1),1); UtempPar = temp; VtempPar = temp;
F11tempPar = temp; F21tempPar = temp; F12tempPar = temp; F22tempPar = temp;
ConvItPerEle = zeros(size(coordinatesFEM,1),1);
markCoordHoleStrainOrNot = zeros(size(coordinatesFEM,1),1);
  
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
            x = [x0temp-winsize/2 ; x0temp+winsize/2 ; x0temp+winsize/2 ; x0temp-winsize/2];  % [coordinates(elements(j,:),1)];
            y = [y0temp-winsize/2 ; y0temp+winsize/2 ; y0temp+winsize/2 ; y0temp-winsize/2];  % [coordinates(elements(j,:),2)];
            tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
            tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
            DfDxImgMaskIndCount = sum(double(1-logical(tempf(:))));
            if DfDxImgMaskIndCount > 0.4 *(winsize+1)^2  % Judge if the window includes area outside the maskfile
                markCoordHoleStrainOrNot(tempj) = 1;
            end
        catch
            markCoordHoleStrainOrNot(tempj) = 1;
        end
        
        try 
            [Utemp, Ftemp, ConvItPerEle(tempj)] = icgn_solver(U0(2*tempj-1:2*tempj), ...
                               x0temp,y0temp,Df,ImgRef,ImgDef,winsize,tol);
            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2);
            F11tempPar(tempj) = Ftemp(1); F21tempPar(tempj) = Ftemp(2); F12tempPar(tempj) = Ftemp(3); F22tempPar(tempj) = Ftemp(4);
        catch
           ConvItPerEle(tempj) = -1;
           UtempPar(tempj) = nan; VtempPar(tempj) = nan;
           F11tempPar(tempj) = nan; F21tempPar(tempj) = nan; F12tempPar(tempj) = nan; F22tempPar(tempj) = nan;
           if showWaitbar, waitbar(tempj/(size(coordinatesFEM,1))); end
       end
    end
    if showWaitbar, close(h); end
    LocalTime = toc;
    
%% ClusterNo > 1: parallel computing
else
    
    % Start parallel computing
    % ****** This step needs to be careful: may be out of memory ******
    disp('--- Set up Parallel pool ---'); tic;
    if showWaitbar, hbar = parfor_progressbar(size(coordinatesFEM,1),'Please wait for Subproblem 1 IC-GN iterations!'); end
    parfor tempj = 1:size(coordinatesFEM,1)  % tempj is the element index
        
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2));  
         
        try
            x = [x0temp-winsize/2 ; x0temp+winsize/2 ; x0temp+winsize/2 ; x0temp-winsize/2];  % [coordinates(elements(j,:),1)];
            y = [y0temp-winsize/2 ; y0temp+winsize/2 ; y0temp+winsize/2 ; y0temp-winsize/2];  % [coordinates(elements(j,:),2)];
            tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
            tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
            DfDxImgMaskIndCount = sum(double(1-logical(tempf(:))));
            if DfDxImgMaskIndCount > 0.4 *(winsize+1)^2
                markCoordHoleStrainOrNot(tempj) = 1;
            end
        catch
            markCoordHoleStrainOrNot(tempj) = 1;
        end
        
        try 
            [Utemp, Ftemp, ConvItPerEle(tempj)] = icgn_solver(U0(2*tempj-1:2*tempj), ...
                    x0temp,y0temp,Df,ImgRef,ImgDef,winsize,tol);
            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2);
            F11tempPar(tempj) = Ftemp(1); F21tempPar(tempj) = Ftemp(2); F12tempPar(tempj) = Ftemp(3); F22tempPar(tempj) = Ftemp(4);
            if showWaitbar, hbar.iterate(1); end
        catch
            ConvItPerEle(tempj) = -1;
            UtempPar(tempj) = nan; VtempPar(tempj) = nan;
            F11tempPar(tempj) = nan; F21tempPar(tempj) = nan; F12tempPar(tempj) = nan; F22tempPar(tempj) = nan;
            if showWaitbar, hbar.iterate(1); end
        end
    end

    if showWaitbar, close(hbar); end 
    LocalTime = toc;
    
end

U = U0; U(1:2:end) = UtempPar; U(2:2:end) = VtempPar;
F = zeros(4*size(coordinatesFEM,1),1); F(1:4:end) = F11tempPar; F(2:4:end) = F21tempPar; F(3:4:end) = F12tempPar; F(4:4:end) = F22tempPar; 
markCoordHoleStrain = find(markCoordHoleStrainOrNot==1);
     

% ------ Clear bad points for Local DIC ------
if isfield(DICpara, 'ICGNMaxIter'), maxIterNum = DICpara.ICGNMaxIter; else, maxIterNum = 100; end
[LocalICGNBadPt, LocalICGNBadPtNum] = detect_bad_points(ConvItPerEle, maxIterNum, coordinatesFEM, 1.0, 6);

nMaskOnly = length(find(ConvItPerEle(:)==maxIterNum+2));
disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-nMaskOnly), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-nMaskOnly)),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN;
F(4*LocalICGNBadPt-3) = NaN; F(4*LocalICGNBadPt-2) = NaN; F(4*LocalICGNBadPt-1) = NaN; F(4*LocalICGNBadPt) = NaN;

% Fill NaN displacements and deformation gradients
U = fill_nan_rbf(U, coordinatesFEM, Df.imgSize, Df.ImgRefMask, 2);
F = fill_nan_rbf(F, coordinatesFEM, Df.imgSize, Df.ImgRefMask, 4);






