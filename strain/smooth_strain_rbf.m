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
try smoothness = DICpara.StrainSmoothness; 
catch smoothness = 1e-5;
end


%% prompt = 'Do you want to smooth displacement? (0-yes; 1-no)';
DoYouWantToSmoothOnceMore = 0; % DoYouWantToSmoothOnceMore = input(prompt);
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
     
    % %dilatedI = ( imgaussfilt(double(DICpara.ImgRefMask),0.5) );
    % dilatedI = logical( dilatedI > 0.01);
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
            
        catch
        end
    end
    
    
    % %     %%%%% Remove outliers %%%%%
    % %     [~,F11RemoveOutlier] = rmoutliers(F(1:4:end), 'movmedian', 1+winstepsize);
    % %     [~,F21RemoveOutlier] = rmoutliers(F(2:4:end), 'movmedian', 1+winstepsize);
    % %     [~,F12RemoveOutlier] = rmoutliers(F(3:4:end), 'movmedian', 1+winstepsize);
    % %     [~,F22RemoveOutlier] = rmoutliers(F(4:4:end), 'movmedian', 1+winstepsize);
    % %     [F11RemoveOutlierInd,~] = find(F11RemoveOutlier==1);
    % %     [F21RemoveOutlierInd,~] = find(F21RemoveOutlier==1);
    % %     [F12RemoveOutlierInd,~] = find(F12RemoveOutlier==1);
    % %     [F22RemoveOutlierInd,~] = find(F22RemoveOutlier==1);
    % % 
    % %     for tempj=1:4
    % %         F(4*F11RemoveOutlierInd-4+tempj) = nan;
    % %         F(4*F21RemoveOutlierInd-4+tempj) = nan;
    % %         F(4*F12RemoveOutlierInd-4+tempj) = nan;
    % %         F(4*F22RemoveOutlierInd-4+tempj) = nan;
    % %     end
    
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
    
    %     Coordxnodes = [min(coordinatesFEM(:,1)):h:max(coordinatesFEM(:,1))]'; 
    %     Coordynodes = [min(coordinatesFEM(:,2)):h:max(coordinatesFEM(:,2))]';
    %     % Iblur_11 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), F(1:4:end),Coordxnodes,Coordynodes,'regularizer','springs'); 
    %     % Iblur_11=Iblur_11';
    %     % Iblur_22 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), F(4:4:end),Coordxnodes,Coordynodes,'regularizer','springs');  
    %     % Iblur_22=Iblur_22';
    %     % Iblur_21 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), F(2:4:end),Coordxnodes,Coordynodes,'regularizer','springs'); 
    %     % Iblur_21=Iblur_21';
    %     % Iblur_12 = gridfit(coordinatesFEM(:,1), coordinatesFEM(:,2), F(3:4:end),Coordxnodes,Coordynodes,'regularizer','springs');
    %     % Iblur_12=Iblur_12';
    %     
    %     Iblur_11 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],F(1:4:end),{Coordxnodes,Coordynodes},smoothness);
    %     Iblur_22 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],F(4:4:end),{Coordxnodes,Coordynodes},smoothness);
    %     Iblur_21 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],F(2:4:end),{Coordxnodes,Coordynodes},smoothness);
    %     Iblur_12 = regularizeNd([coordinatesFEM(:,1), coordinatesFEM(:,2)],F(3:4:end),{Coordxnodes,Coordynodes},smoothness);
    %     
    %     % -------------------------------------------------------
    %     imageFilter=fspecial('gaussian',FilterSizeInput,FilterStd);
    %     Iblur_1 = nanconv(Iblur_11,imageFilter,'edge','nanout');
    %     Iblur_4 = nanconv(Iblur_22,imageFilter,'edge','nanout');
    %     Iblur_2 = nanconv(Iblur_21,imageFilter,'edge','nanout');
    %     Iblur_3 = nanconv(Iblur_12,imageFilter,'edge','nanout');
    %     
    %     for tempi = 1:size(coordinatesFEM,1)
    %         [row1,~] = find(Coordxnodes==coordinatesFEM(tempi,1));
    %         [row2,~] = find(Coordynodes==coordinatesFEM(tempi,2));
    %         F(4*tempi-3) = Iblur_1(row1,row2);
    %         F(4*tempi)   = Iblur_4(row1,row2);
    %         F(4*tempi-2) = Iblur_2(row1,row2);
    %         F(4*tempi-1) = Iblur_3(row1,row2);
    %     end
    %      
    % prompt = 'Do you want to smooth displacement once more? (0-yes; 1-no)';
    % DoYouWantToSmoothOnceMore = input(prompt); 
    
    
    SmoothTimes = SmoothTimes+1;
    if SmoothTimes > 1
        DoYouWantToSmoothOnceMore = 1;
    end
    
end

