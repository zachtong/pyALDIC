function F = smooth_strain_rbf(F,DICmesh,DICpara)
%smooth_strain_rbf: to smooth solved strain fields by curvature regularization
% 	F = smooth_strain_rbf(F,DICmesh,DICpara)
% ----------------------------------------------
%
%   INPUT: F                 Deformation gradient tensor: 
%                            F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%          DICmesh           DIC mesh
%          DICpara           DIC parameters
%
%   OUTPUT: F                Smoothed strain fields by curvature regularization
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
% Last time updated: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
h = DICmesh.elementMinSize;
winstepsize = DICpara.winstepsize;
coordinatesFEM = DICmesh.coordinatesFEM;
FilterSizeInput = DICpara.StrainFilterSize;
FilterStd = DICpara.StrainFilterStd; 
F = full(F); 
smoothness = DICpara.StrainSmoothness;


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
        
        try % if stats(tempi).Area > 20
            
            %%%%% Find those nodes %%%%%
            indPxtempi = sub2ind( DICpara.ImgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
            Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
            
            % ------ F11 ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[F(4*LiaList-3)]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            F(4*LiaList-3) = fi1(:);
            
            % ------ F21 ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[F(4*LiaList-2)]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            F(4*LiaList-2) = fi1(:);
            
            % ------ F12 ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[F(4*LiaList-1)]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            F(4*LiaList-1) = fi1(:);
            
            % ------ F22 ------
            op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[F(4*LiaList )]','RBFFunction', 'thinplate', 'RBFSmooth',smoothness); % rbfcheck(op1);
            fi1 = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
            F(4*LiaList ) = fi1(:);
            
        catch ME
            warning('smooth_strain_rbf:rbfFit', 'RBF strain smoothing failed for region %d: %s', tempi, ME.message);
        end
    end

    %%%%%% Fill nans %%%%%%
    nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);
    
    if   ~isempty(nanindexF)
        
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-3),'nearest','nearest');
        F11 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-2),'nearest','nearest');
        F21 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-1),'nearest','nearest');
        F12 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),F(4*notnanindexF-0),'nearest','nearest');
        F22 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
        
        F = [F11(:),F21(:),F12(:),F22(:)]'; F = F(:);

    end

    SmoothTimes = SmoothTimes+1;
    if SmoothTimes > 1
        DoYouWantToSmoothOnceMore = 1;
    end
    
end

