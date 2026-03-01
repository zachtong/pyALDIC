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
%ConvItPerEleMean = median(ConvItPerEle(LocalICGNGoodPt));
%ConvItPerEleStd = std(ConvItPerEle(LocalICGNGoodPt));
pd = fitdist( ConvItPerEle(LocalICGNGoodPt), 'Normal'  );
ConvItPerEleMean = pd.mu;
ConvItPerEleStd = pd.sigma;
[row4,~] =  find(ConvItPerEle(:) > max([ConvItPerEleMean+1*ConvItPerEleStd, 6 ])); % Here "0.15" is an empirical value
LocalICGNBadPt = unique(union(LocalICGNBadPt,row4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Print results info on the MATLAB command window
disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-length(row3)), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3))),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN; 
F(4*LocalICGNBadPt-3) = NaN; F(4*LocalICGNBadPt-2) = NaN; F(4*LocalICGNBadPt-1) = NaN; F(4*LocalICGNBadPt) = NaN;
 
% figure,plot3(coordinatesFEM(:,1),coordinatesFEM(:,2),F(1:4:end),'.')
    

%%%%%% Fill nans %%%%%%
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);


% dilatedI = ( imgaussfilt(double(Df.ImgRefMask),0.5) );
% dilatedI = logical( dilatedI > 0.5); % figure, imshow(dilatedI)
dilatedI = Df.ImgRefMask; 
cc = bwconncomp(dilatedI,8);
indPxAll = sub2ind( Df.imgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)) );
indPxNotNanAll = sub2ind( Df.imgSize, round(coordinatesFEM(notnanindex,1)), round(coordinatesFEM(notnanindex,2)) );
stats = regionprops(cc,'Area','PixelList'); 
for tempi = 1:length(stats)
    
   try %if stats(tempi).Area > 20
        
    %%%%% Find those nodes %%%%%
    indPxtempi = sub2ind( Df.imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
    Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
    Lib = ismember(indPxNotNanAll,indPxtempi); [LibList,~] = find(Lib==1);
    
    % %%%%% RBF (Radial basis function) works better than "scatteredInterpolant" %%%%%
    % ------ Disp u ------
    op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList)-1)]','RBFFunction', 'thinplate' ); %rbfcheck(op1);
    fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
    U(2*LiaList-1) = fi1(:);
    % figure, plot3(coordinatesFEM(notnanindex(LibList),1),coordinatesFEM(notnanindex(LibList),2),U(2*notnanindex(LibList)-1),'.')
    % hold on; plot3(coordinatesFEM(LiaList,1),coordinatesFEM(LiaList,2),fi1,'.')
    
    % ------ Disp v ------
    op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList) )]','RBFFunction', 'thinplate' ); %rbfcheck(op1);
    fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
    U(2*LiaList ) = fi1(:);
    
    
    % if  (LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3)) < 0.4
    
        % ------ F11 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-3)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-3) = fi1(:);
        % ------ F21 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-2)]','RBFFunction', 'thinplate');% rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-2) = fi1(:);
        % ------ F12 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-1)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList-1) = fi1(:);
        % ------ F22 ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[F(4*notnanindex(LibList)-0)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        F(4*LiaList) = fi1(:);
    
    % end
    
    catch ME
        warning('local_icgn:nanFill', 'RBF NaN fill failed for region %d: %s', tempi, ME.message);
    end
end


%% Final NaN filling (scatteredInterpolant fallback)
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);

if ~isempty(nanindex) || ~isempty(nanindexF)

Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'natural','nearest');
U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'natural','nearest');
U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-3),'natural','nearest');
F11 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-2),'natural','nearest');
F21 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-1),'natural','nearest');
F12 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-0),'natural','nearest');
F22 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

U = [U1(:),U2(:)]'; U = U(:);
F = [F11(:),F21(:),F12(:),F22(:)]'; F = F(:);

end






