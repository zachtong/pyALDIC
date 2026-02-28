function [U,HPar,ALSub1Time,ConvItPerEle,LocalICGNBadPtNum] = subpb1_solver(UOld,FOld,udual,vdual,coordinatesFEM,...
                                            Df,ImgRef,ImgDef,mu,beta,HPar,ALSolveStep,DICpara,ICGNmethod,tol)
%FUNCTION [U,HPar,ALSub1Time,ConvItPerEle,LocalICGNBadPtNum] = subpb1_solver(U,F,udual,vdual,coordinatesFEM,...
%                                            Df,ImgRef,ImgDef,mu,beta,HPar,ALSolveStep,DICpara,ICGNmethod,tol)
% The ALDIC Subproblem 1 ICGN subset solver (part I): to assign a sequential or a parallel computing
% (see part II: ./func/funICGN_Subpb1.m)
% ----------------------------------------------
%   INPUT: UOld                 Initial guess of the displacement fields
%          FOld                 Initial guess of the deformation gradients
%          udual,vdual          Dual variables
%          coordinatesFEM       FE mesh coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          mu,beta              ALDIC coefficients
%          HPar                 Stored Hessian matrix
%          ALSolveStep          ALDIC ADMM iteration step #
%          DICpara              DIC parameters: subset size, subset spacing, ...
%          ICGNmethod           ICGN iteration scheme: 'GaussNewton' -or- 'LevenbergMarquardt'
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           HPar                Hessian matrix for each local subset
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
warning('off');
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
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2)); HLocal = zeros(6,6);
        if ALSolveStep > 1
            HLocal(1:6) = [HPar{1}(tempj); HPar{2}(tempj); HPar{3}(tempj); HPar{4}(tempj); HPar{5}(tempj); HPar{6}(tempj)];
            HLocal(8:12) = [HPar{7}(tempj); HPar{8}(tempj); HPar{9}(tempj); HPar{10}(tempj); HPar{11}(tempj)];
            HLocal(15:18) = [HPar{12}(tempj); HPar{13}(tempj); HPar{14}(tempj); HPar{15}(tempj)];
            HLocal(22:24) = [HPar{16}(tempj); HPar{17}(tempj); HPar{18}(tempj)];
            HLocal(29:30) = [HPar{19}(tempj); HPar{20}(tempj)]; HLocal(36) = HPar{21}(tempj);
            HLocal = HLocal'+HLocal-diag(diag(HLocal));
        end
        
        try
            [Utemp,~,ConvItPerEle(tempj,:)] = icgn_subpb1( ...
                x0temp,y0temp,Df,ImgRef,ImgDef,winsize_x(tempj),winsize_y(tempj),...
                HLocal,beta,mu,udual(4*tempj-3:4*tempj),vdual(2*tempj-1:2*tempj),...
                UOld(2*tempj-1:2*tempj),FOld(4*tempj-3:4*tempj),tol,ICGNmethod);
            
            % disp(['ele ',num2str(tempj),' converge or not is ',num2str(ConvItPerEle(tempj,:)),' (1-converged; 0-unconverged)']);
            
            % Store solved deformation gradients
            UPar{1}(tempj) = Utemp(1); UPar{2}(tempj) = Utemp(2);
        catch
            UPar{1}(tempj) = nan; UPar{2}(tempj) = nan;
            Htemp = zeros(6,6);
        end
        
        if ALSolveStep == 1
            HPar{1}(tempj) = Htemp(1); HPar{2}(tempj) = Htemp(2); HPar{3}(tempj) = Htemp(3); HPar{4}(tempj) = Htemp(4); HPar{5}(tempj) = Htemp(5);
            HPar{6}(tempj) = Htemp(6); HPar{7}(tempj) = Htemp(8); HPar{8}(tempj) = Htemp(9); HPar{9}(tempj) = Htemp(10); HPar{10}(tempj) = Htemp(11);
            HPar{11}(tempj) = Htemp(12); HPar{12}(tempj) = Htemp(15); HPar{13}(tempj) = Htemp(16); HPar{14}(tempj) = Htemp(17);
            HPar{15}(tempj) = Htemp(18); HPar{16}(tempj) = Htemp(22); HPar{17}(tempj) = Htemp(23); HPar{18}(tempj) = Htemp(24);
            HPar{19}(tempj) = Htemp(29); HPar{20}(tempj) = Htemp(30); HPar{21}(tempj) = Htemp(36);
        end
        % waitbar(tempj/(size(coordinatesFEM,1)));
    end
    if showWaitbar, close(h); end
    ALSub1Time = toc;
    

%% ClusterNo > 1: parallel computing
else
    
    % Start parallel computing
    % ****** This step needs to be careful: may be out of memory ******
    % delete(gcp);parpool(ClusterNo); tic;
    if showWaitbar, hbar = parfor_progressbar(size(coordinatesFEM,1),'Please wait for Subproblem 1 IC-GN iterations!'); end
    HPar1 = HPar{1}; HPar2 = HPar{2}; HPar3 = HPar{3}; HPar4 = HPar{4}; HPar5 = HPar{5}; HPar6 = HPar{6}; HPar7 = HPar{7};
    HPar8 = HPar{8}; HPar9 = HPar{9}; HPar10 = HPar{10}; HPar11 = HPar{11}; HPar12 = HPar{12}; HPar13 = HPar{13}; HPar14 = HPar{14};
    HPar15 = HPar{15}; HPar16 = HPar{16}; HPar17 = HPar{17}; HPar18 = HPar{18}; HPar19 = HPar{19}; HPar20 = HPar{20}; HPar21 = HPar{21};
    UtempPar = UPar{1}; VtempPar = UPar{2};
    
    parfor tempj = 1:size(coordinatesFEM,1)
        x0temp = round(coordinatesFEM(tempj,1)); y0temp = round(coordinatesFEM(tempj,2)); HLocal = zeros(6,6);
        if ALSolveStep > 1
            HLocal(1:6) = [HPar1(tempj);HPar2(tempj);HPar3(tempj);HPar4(tempj);HPar5(tempj);HPar6(tempj)];
            HLocal(8:12) = [HPar7(tempj);HPar8(tempj);HPar9(tempj);HPar10(tempj);HPar11(tempj)];
            HLocal(15:18) = [HPar12(tempj);HPar13(tempj);HPar14(tempj);HPar15(tempj)];
            HLocal(22:24) = [HPar16(tempj);HPar17(tempj);HPar18(tempj)];
            HLocal(29:30) = [HPar19(tempj);HPar20(tempj)]; HLocal(36) = HPar21(tempj);
            HLocal = HLocal'+HLocal-diag(diag(HLocal));
        end
        
       try
            [Utemp,~,ConvItPerEle(tempj,:)] = icgn_subpb1( ...
                x0temp,y0temp,Df,ImgRef,ImgDef,winsize_x(tempj),winsize_y(tempj),...
                HLocal,beta,mu,udual(4*tempj-3:4*tempj),vdual(2*tempj-1:2*tempj),...
                UOld(2*tempj-1:2*tempj),FOld(4*tempj-3:4*tempj),tol,ICGNmethod);
            
            % disp(['ele ',num2str(tempj),' converge or not is ',num2str(ConvItPerEle(tempj,:)),' (1-converged; 0-unconverged)']);
            
            % Store solved deformation gradients
            UtempPar(tempj) = Utemp(1); VtempPar(tempj) = Utemp(2);
        catch
            UtempPar(tempj) = nan; VtempPar(tempj) = nan;
            Htemp = zeros(6,6);
        end
        if ALSolveStep == 1
            HPar1(tempj) = Htemp(1); HPar2(tempj) = Htemp(2); HPar3(tempj) = Htemp(3); HPar4(tempj) = Htemp(4); HPar5(tempj) = Htemp(5);
            HPar6(tempj) = Htemp(6); HPar7(tempj) = Htemp(8); HPar8(tempj) = Htemp(9); HPar9(tempj) = Htemp(10); HPar10(tempj) = Htemp(11);
            HPar11(tempj) = Htemp(12); HPar12(tempj) = Htemp(15); HPar13(tempj) = Htemp(16); HPar14(tempj) = Htemp(17);
            HPar15(tempj) = Htemp(18); HPar16(tempj) = Htemp(22); HPar17(tempj) = Htemp(23); HPar18(tempj) = Htemp(24);
            HPar19(tempj) = Htemp(29); HPar20(tempj) = Htemp(30); HPar21(tempj) = Htemp(36);
        end
        if showWaitbar, hbar.iterate(1); end
    end
    if showWaitbar, close(hbar); end
    ALSub1Time = toc;
    HPar{1} = HPar1; HPar{2} = HPar2; HPar{3} = HPar3; HPar{4} = HPar4; HPar{5} = HPar5; HPar{6} = HPar6; HPar{7} = HPar7;
    HPar{8} = HPar8; HPar{9} = HPar9; HPar{10} = HPar10; HPar{11} = HPar11; HPar{12} = HPar12; HPar{13} = HPar13; HPar{14} = HPar14;
    HPar{15} = HPar15; HPar{16} = HPar16; HPar{17} = HPar17; HPar{18} = HPar18; HPar{19} = HPar19; HPar{20} = HPar20; HPar{21} = HPar21;
    UPar{1} = UtempPar; UPar{2} = VtempPar;
    
    % clear HPar1 HPar2 HPar3 HPar4 HPar5 HPar6 HPar7 HPar8 HPar9 HPar10 HPar11 HPar12 HPar13 HPar14 HPar15 HPar16 HPar17 HPar18 HPar19 HPar20 HPar21
end

U = UOld(:);
fprintf('[debug Subpb1] After UOld(:): U=%s, UPar1=%s, UPar2=%s\n', mat2str(size(U)), mat2str(size(UPar{1})), mat2str(size(UPar{2})));
U(1:2:end) = UPar{1}; U(2:2:end) = UPar{2};
fprintf('[debug Subpb1] After UPar assign: U=%s\n', mat2str(size(U)));

% ------ Clear bad points for Local DIC ------
% find bad points after Local Subset ICGN
maxIterNum = 100;
[row1,~] = find(ConvItPerEle(:)<0);
[row2,~] = find(ConvItPerEle(:)>maxIterNum-1);
[row3,~] = find(ConvItPerEle(:)==maxIterNum+2);
LocalICGNBadPt = unique(union(row1,row2)); LocalICGNBadPtNum = length(LocalICGNBadPt)-length(row3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Though some subsets are converged, but their accuracy is worse than most
% % other subsets. This step is to remove those subsets with abnormal convergence steps
LocalICGNGoodPt = setdiff([1:1:size(coordinatesFEM,1)],LocalICGNBadPt);
ConvItPerEleMean = mean(ConvItPerEle(LocalICGNGoodPt));
ConvItPerEleStd = std(ConvItPerEle(LocalICGNGoodPt));
[row4,~] = find(ConvItPerEle(:) > max([ConvItPerEleMean+0.25*ConvItPerEleStd, 10])); % Here "0.25" is an empirical value
LocalICGNBadPt = unique(union(LocalICGNBadPt,row4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['Local ICGN bad subsets %: ', num2str(LocalICGNBadPtNum),'/',num2str(size(coordinatesFEM,1)-length(row3)), ...
    '=',num2str(100*(LocalICGNBadPtNum)/(size(coordinatesFEM,1)-length(row3))),'%']);
U(2*LocalICGNBadPt-1) = NaN; U(2*LocalICGNBadPt) = NaN;
fprintf('[debug Subpb1] After bad pt NaN: U=%s, nBadPt=%d, nNaN_in_U=%d\n', mat2str(size(U)), length(LocalICGNBadPt), sum(isnan(U)));

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
        % figure, plot3(coordinatesFEM(notnanindex(LibList),1),coordinatesFEM(notnanindex(LibList),2),U(2*notnanindex(LibList)-1),'.')
        % hold on; plot3(coordinatesFEM(LiaList,1),coordinatesFEM(LiaList,2),fi1,'.')

        % ------ Disp v ------
        op1 = rbfcreate( round([coordinatesFEM(notnanindex(LibList),1:2)]'),[U(2*notnanindex(LibList) )]','RBFFunction', 'thinplate'); rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        U(2*LiaList ) = fi1(:);
    
    catch
    end
     
end
fprintf('[debug Subpb1] After RBF fill: U=%s, nNaN=%d\n', mat2str(size(U)), sum(isnan(U)));

%%%%% Remove outliers %%%%%
% [~,URemoveOutlier] = rmoutliers(U(1:2:end), 'movmedian', 1+winstepsize);
% [~,VRemoveOutlier] = rmoutliers(U(2:2:end), 'movmedian', 1+winstepsize);
% [URemoveOutlierInd,~] = find(URemoveOutlier==1);
% [VRemoveOutlierInd,~] = find(VRemoveOutlier==1);
% 
% for tempj=1:2
%     F(2*URemoveOutlierInd-4+tempj) = nan;
%     F(2*VRemoveOutlierInd-4+tempj) = nan;
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Fill nans %%%%%%
nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
fprintf('[debug Subpb1] Before final fill: U=%s, nanindex=%d, notnanindex=%d\n', mat2str(size(U)), length(nanindex), length(notnanindex));

if ~isempty(nanindex) && ~isempty(notnanindex)

    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'natural','nearest');
    U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'natural','nearest');
    U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

    U = [U1(:),U2(:)]'; U = U(:);
    fprintf('[debug Subpb1] After final fill: U=%s, U1=%s, U2=%s\n', mat2str(size(U)), mat2str(size(U1)), mat2str(size(U2)));
elseif ~isempty(nanindex) && isempty(notnanindex)
    % All nodes are NaN (all ICGN calls failed) — fall back to previous result
    warning('subpb1_solver:allNaN', 'All ICGN nodes failed, falling back to UOld.');
    U = UOld(:);
end
fprintf('[debug Subpb1] Return U=%s\n', mat2str(size(U)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== Use gridfit to interpolate ======
% Coordxnodes = unique(coordinatesFEM(:,1)); Coordynodes = unique(coordinatesFEM(:,2));
%
% [u1temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), U(2*notnanindex-1),Coordxnodes,Coordynodes,'regularizer','springs'); u1temp = u1temp';
% [v1temp] = gridfit(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), U(2*notnanindex),Coordxnodes,Coordynodes,'regularizer','springs'); v1temp = v1temp';
%   
% % Add remove outliers median-test based on:
% % u1=cell(2,1); u1{1}=u1temp; u1{2}=v1temp;
% % [u2] = removeOutliersMedian(u1,4); u2temp=u2{1}; v2temp=u2{2};
% for tempi = 1:size(coordinatesFEM,1)
%     [row1,col1] = find(Coordxnodes==coordinatesFEM(tempi,1));
%     [row2,col2] = find(Coordynodes==coordinatesFEM(tempi,2));
%     U(2*tempi-1) = u1temp(row1,row2);
%     U(2*tempi)   = v1temp(row1,row2); 
% end





