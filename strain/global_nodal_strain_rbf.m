function [StrainNodalPt] = global_nodal_strain_rbf(DICmesh,DICpara,U)
%global_nodal_strain_rbf:  to compute strain fields by the RBF interpolation
%   [StrainNodalPt] = funGlobalNodalStrainRBF(DICmesh,U)                
% ----------------------------------------------
%
%	INPUT: DICmesh             DIC FE Q4 mesh: coordinatesFEM, elementsFEM
%          U                   Disp vector: U = [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           
%   OUTPUT: StrainNodalPt      Solved strains at the nodal points
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
% Last time updated: 2018.03, 2020.12 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
coordinatesFEM = DICmesh.coordinatesFEM; 
winstepsize = DICpara.winstepsize;
SizeCoords = size(coordinatesFEM,1);
if isfield(DICpara, 'strainRBFSmoothness')
    smoothness = DICpara.strainRBFSmoothness;
else
    smoothness = 1e-3;
end

 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
StrainNodalPt = NaN(4*size(coordinatesFEM,1),1);

% dilatedI = ( imgaussfilt(double(DICpara.ImgRefMask),0.5) );
% dilatedI = logical( dilatedI > 0.01);
dilatedI = logical(DICpara.ImgRefMask); % figure, imshow(dilatedI);
cc = bwconncomp(dilatedI,8);
indPxAll = sub2ind( DICpara.ImgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)) );
 
stats = regionprops(cc,'Area','PixelList');
for tempi = 1:length(stats)
    
    if stats(tempi).Area > 20
        
        %%%%% Find those nodes %%%%%
        indPxtempi = sub2ind( DICpara.ImgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
        Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
         
        % ------ du ------
        op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[U(2*LiaList-1)]','RBFFunction', 'thinplate','RBFSmooth',smoothness); % rbfcheck(op1);
        [~,dudx,dudy] = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        StrainNodalPt(4*LiaList-3) = dudx(:);
        StrainNodalPt(4*LiaList-1) = dudy(:);
        
        % ------ dv ------
        op1 = rbfcreate( [coordinatesFEM(LiaList,1:2)]',[U(2*LiaList )]','RBFFunction', 'thinplate','RBFSmooth',smoothness); % rbfcheck(op1);
        [~,dvdx,dvdy] = rbfinterp( [coordinatesFEM(LiaList,1:2)]', op1);
        StrainNodalPt(4*LiaList-2) = dvdx(:);
        StrainNodalPt(4*LiaList-0) = dvdy(:);
        
    end
end

%%%%% Remove outliers %%%%%
[~,F11RemoveOutlier] = rmoutliers(StrainNodalPt(1:4:end), 'movmedian', 1+winstepsize);
[~,F21RemoveOutlier] = rmoutliers(StrainNodalPt(2:4:end), 'movmedian', 1+winstepsize);
[~,F12RemoveOutlier] = rmoutliers(StrainNodalPt(3:4:end), 'movmedian', 1+winstepsize);
[~,F22RemoveOutlier] = rmoutliers(StrainNodalPt(4:4:end), 'movmedian', 1+winstepsize);
[F11RemoveOutlierInd,~] = find(F11RemoveOutlier==1);
[F21RemoveOutlierInd,~] = find(F21RemoveOutlier==1);
[F12RemoveOutlierInd,~] = find(F12RemoveOutlier==1);
[F22RemoveOutlierInd,~] = find(F22RemoveOutlier==1);

for tempj=1:4
    StrainNodalPt(4*F11RemoveOutlierInd-4+tempj) = nan;
    StrainNodalPt(4*F21RemoveOutlierInd-4+tempj) = nan;
    StrainNodalPt(4*F12RemoveOutlierInd-4+tempj) = nan;
    StrainNodalPt(4*F22RemoveOutlierInd-4+tempj) = nan;
end

%%%%%% Fill nans %%%%%%
nanindexF = find(isnan(StrainNodalPt(1:4:end))==1); notnanindexF = setdiff([1:1:size(coordinatesFEM,1)],nanindexF);

if  ~isempty(nanindexF)
    
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-3),'linear','linear');
    F11 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-2),'linear','linear');
    F21 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-1),'linear','linear');
    F12 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-0),'linear','linear');
    F22 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));


    StrainNodalPt = [F11(:),F21(:),F12(:),F22(:)]'; StrainNodalPt = StrainNodalPt(:);

end


 
 