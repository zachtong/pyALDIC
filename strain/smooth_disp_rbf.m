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
try smoothness = DICpara.DispSmoothness; 
catch smoothness = 5e-4;
end


%% prompt = 'Do you want to smooth displacement? (0-yes; 1-no)';
DoYouWantToSmoothOnceMore = 0; % DoYouWantToSmoothOnceMore = input(prompt);
if DoYouWantToSmoothOnceMore == 0  
    if isempty(FilterStd) == 1
        prompt = 'Choose filter standard deviation(0-default): ';
        FilterStd = input(prompt);
        if FilterStd == 0
            FilterStd = 0.5; 
        end
    else
        if FilterStd == 0
            FilterStd = 0.5;
        end
    end
    if isempty(FilterSizeInput) == 1
        prompt = 'Choose Gaussian filter size(0-default): ';
        FilterSizeInput = input(prompt);
        if FilterSizeInput == 0
            FilterSizeInput = 2*ceil(2*FilterStd)+1; 
        end
    else
        if FilterSizeInput == 0
            FilterSizeInput = 2*ceil(2*FilterStd)+1;
        end
    end
end
 
SmoothTimes = 1;
while (DoYouWantToSmoothOnceMore==0)
     
    % dilatedI = ( imgaussfilt(double(DICpara.ImgRefMask),0.5) );
    % dilatedI = logical( dilatedI > 0.01);
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
    %     Coordxnodes = [min(coordinatesFEM(:,1)):h:max(coordinatesFEM(:,1))]'; 
    %     Coordynodes = [min(coordinatesFEM(:,2)):h:max(coordinatesFEM(:,2))]';
    %     %Iblur_10 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), U(1:2:end),Coordxnodes,Coordynodes, 'regularizer','springs'); Iblur_10=Iblur_10';
    %     %Iblur_20 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), U(2:2:end),Coordxnodes,Coordynodes,  'regularizer','springs'); Iblur_20=Iblur_20';
    %     Iblur_10 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],U(1:2:end),{Coordxnodes,Coordynodes},smoothness);
    %     Iblur_20 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],U(2:2:end),{Coordxnodes,Coordynodes},smoothness);
    %     
    %     imageFilter=fspecial('gaussian',FilterSizeInput,FilterStd);
    %      
    %     Iblur_1 = nanconv(Iblur_10,imageFilter,'edge','nanout');
    %     Iblur_2 = nanconv(Iblur_20,imageFilter,'edge','nanout');
    %     for tempi = 1:size(coordinatesFEM,1)
    %         [row1,~] = find(Coordxnodes==coordinatesFEM(tempi,1));
    %         [row2,~] = find(Coordynodes==coordinatesFEM(tempi,2));
    %         U(2*tempi-1) = Iblur_1(row1,row2);
    %         U(2*tempi)   = Iblur_2(row1,row2);
    %     end
     
    % prompt = 'Do you want to smooth displacement once more? (0-yes; 1-no)';
    % DoYouWantToSmoothOnceMore = input(prompt); 
    SmoothTimes = SmoothTimes+1;
    if SmoothTimes > 1
        DoYouWantToSmoothOnceMore = 1;
    end
    
end


