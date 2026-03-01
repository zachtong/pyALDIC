function U = smooth_disp_rbf(U,DICmesh,DICpara)
%FUNCTION U = smooth_disp_rbf(U,DICmesh,DICpara)
% Object: Smooth solved displacement fields by curvature regularization
% ----------------------------------------------
%
%   INPUT: U                 Displacement vector: U = [Ux_node1, Uy_node1, Ux_node2, Uy_node2, ... , Ux_nodeN, Uy_nodeN]';
%          DICmesh           DIC mesh
%          DICpara           DIC parameters
%
%   OUTPUT: U                Smoothed displacement fields by curvature regularization
%
% ----------------------------------------------
% Reference
% [1] RegularizeNd. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/61436-regularizend
% [2] Gridfit. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 12/2020.
% ==============================================


%% Initialization
h = DICmesh.elementMinSize;
coordinatesFEM = DICmesh.coordinatesFEM;
FilterSizeInput = DICpara.DispFilterSize;
FilterStd = DICpara.DispFilterStd;
U = full(U);
smoothness = DICpara.DispSmoothness;


DoYouWantToSmoothOnceMore = 0;
if DoYouWantToSmoothOnceMore == 0  
    if isempty(FilterStd) || FilterStd == 0
        FilterStd = 0.5;
    end
    if isempty(FilterSizeInput) || FilterSizeInput == 0
        FilterSizeInput = 2*ceil(2*FilterStd)+1;
    end
end
 
SmoothTimes = 1;
while (DoYouWantToSmoothOnceMore==0)
     
    dilatedI = logical(DICpara.ImgRefMask);
    cc = bwconncomp(dilatedI,8);
    indPxAll = sub2ind( DICpara.ImgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)) );
    
    stats = regionprops(cc,'Area','PixelList');
    for tempi = 1:length(stats)
        
        if stats(tempi).Area > 20
            
            %%%%% Find those nodes %%%%%
            indPxtempi = sub2ind( DICpara.ImgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
            Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
            
            % ------ disp u ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[U(2*LiaList-1)]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            U(2*LiaList-1) = fi1(:);
            
            % ------ disp v ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[U(2*LiaList )]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            U(2*LiaList) = fi1(:);
            
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% Fill nans %%%%%%
    nanindex = find(isnan(U(1:2:end))==1); notnanindex = setdiff([1:1:size(coordinatesFEM,1)],nanindex);
    
    if ~isempty(nanindex)
        
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'nearest','nearest');
        U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'nearest','nearest');
        U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        
        U = [U1(:),U2(:)]'; U = U(:);
    end

    SmoothTimes = SmoothTimes+1;
    if SmoothTimes > 1
        DoYouWantToSmoothOnceMore = 1;
    end
    
end


