function [Df] = img_gradient(ImgRef,ImgDef,varargin)
%FUNCTION [Df] = img_gradient(fNormalized,gNormalized,varargin)
% This function is to compuate image grayscale value gradients 
%
%   INPUT: ImgRef           Reference image grayscale value matrix
%          ImgDef           Not needed anymore
%          ImgRefMask       A binary image mask
%          method           Finite difference (default) or 'splinecubic'
%
%   OUTPUT: Df.Axis         Chosen region of interest (ROI) 
%           Df.DfDx         x-derivatives of image grayscale values
%           Df.DfDy         y-derivatives of image grayscale values
%           Df.ImgRefMask   Used binary image mask
%            
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================


fprintf('\n'); disp('--- Start to compute image gradients ---');
imgSize = size(ImgRef);


%% Read image mask
if nargin > 2
    ImgRefMask = varargin{1};
else
    ImgRefMask = ones(imgSize(1),imgSize(2));
end

if nargin > 3
    method = varargin{2};
else
    method = [];
end

    
%%
if strcmp(method,'splinecubic')
    
    imgfNormalizedbc = Spline2D('bicubic',[1:1:size(fNormalized,1)],[1:1:size(fNormalized,2)],fNormalized);
    imggNormalizedbc = Spline2D('bicubic',[1:1:size(gNormalized,1)],[1:1:size(gNormalized,2)],gNormalized);
    % [XX,YY] = ndgrid([1:1:size(fNormalized,1)],[1:1:size(fNormalized,2)]);
    % DfDxNormalizedbc = imgfNormalizedbc.eval_Dx(XX,YY);
    % DfDyNormalizedbc = imgfNormalizedbc.eval_Dy(XX,YY);
    DfAxis = [0,size(fNormalized,1)-1,0,size(fNormalized,2)-1];
    
else
    
    % Construct finite difference operator
    imgradientMatrix = [-1/60 3/20 -3/4 0 3/4 -3/20 1/60]';
    DfDxStartx = 4; % max([4,coordinatesFEM(1,1)-0.5*winsize-1]);
    DfDxStarty = 4; % max([4,coordinatesFEM(1,2)-0.5*winsize-1]);
    DfDxEndx = size(ImgRef,1)-3; % min([coordinatesFEM(M*N,1)+0.5*winsize,size(fNormalized,1)-3]);
    DfDxEndy = size(ImgRef,2)-3; % min([coordinatesFEM(M*N,2)+0.5*winsize,size(fNormalized,2)-3]);
    
    % Apply finite difference filter
    I = ImgRef( DfDxStartx-3:DfDxEndx+3, DfDxStarty-3:DfDxEndy+3 ); 
    DfDxNormalizedtemp = imfilter(I, imgradientMatrix);
    DfDyNormalizedtemp = imfilter(I, imgradientMatrix');
    
    % Because of the "imfilter", we need to crop +/-3 data points
    DfDxNormalized = DfDxNormalizedtemp(4:end-3, 4:end-3); 
    DfDyNormalized = DfDyNormalizedtemp(4:end-3, 4:end-3);
    
    % Assemble data set
    DfAxis = [DfDxStartx,DfDxEndx,DfDxStarty,DfDxEndy];
    Df.DfCropWidth = 3;
    Df.DfAxis = DfAxis; Df.imgSize = imgSize;
    Df.DfDx = DfDxNormalized;
    Df.DfDy = DfDyNormalized;
    
    % Multiply the image mask
    Df.ImgRefMask = ImgRefMask; % Generate image mask
    Df.DfDx = (Df.DfDx) .* Df.ImgRefMask(Df.DfAxis(1):Df.DfAxis(2), Df.DfAxis(3):Df.DfAxis(4));
    Df.DfDy = (Df.DfDy) .* Df.ImgRefMask(Df.DfAxis(1):Df.DfAxis(2), Df.DfAxis(3):Df.DfAxis(4));
     
    % figure, surf(Df.DfDx,'edgecolor','none')
    % view([90,90]); axis equal; axis tight; colormap gray; cb=colorbar; caxis([-0.5,0.5]);
    
end

disp('--- Computing image gradients done ---');


end
