function [mask_file_name,ImgMask] = read_masks(varargin)
%FUNCTION [file_name,Img,DICpara] = read_masks(varargin)
% ----------------------------------------------
%   This script is to load DIC images 
%   Images can be loaded by:
%       i) selecting a folder which included all the DIC raw images, 
%       ii) inputing image file name prefix keywords
%       iii) manually select DIC raw images
%
%   INPUT: No inputs are needed
%
%   OUTPUT: file_name    Loaded DIC raw image file name
%           Img          Loaded DIC images
%           DICpara      DIC parameters
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 12/2020.
% ==============================================

%%
fprintf('Choose method to load image mask files:  \n')
fprintf('     0: Select images mask file folder;  \n')
fprintf('     1: Use prefix of image mask file names;  \n')
fprintf('     2: Manually select image mask files.  \n')
prompt = 'Input here: ';
LoadImgMethod = input(prompt);

switch LoadImgMethod 
    case 0
        % ==============================================
        imgfoldername = uigetdir(pwd,'Select image mask file folder');
        addpath([imgfoldername,'\']);
        img1 = dir(fullfile(imgfoldername,'*.jpg'));
        img2 = dir(fullfile(imgfoldername,'*.jpeg'));
        img3 = dir(fullfile(imgfoldername,'*.tif'));
        img4 = dir(fullfile(imgfoldername,'*.tiff'));
        img5 = dir(fullfile(imgfoldername,'*.bmp'));
        img6 = dir(fullfile(imgfoldername,'*.png'));
        img7 = dir(fullfile(imgfoldername,'*.jp2'));
        mask_file_name = [img1;img2;img3;img4;img5;img6;img7];
        mask_file_name = struct2cell(mask_file_name);
    case 1
        % ==============================================
        fprintf('What is prefix of DIC images? E.g. img_0*.tif.   \n')
        prompt = 'Input here: ';
        mask_file_name = input(prompt,'s');
        [~,imgname,imgext] = fileparts(mask_file_name);
        mask_file_name = dir([imgname,imgext]);
        mask_file_name = struct2cell(mask_file_name);
    otherwise
        % ==============================================
        disp('--- Please load first image mask file ---')
        mask_file_name{1,1} = uigetfile('*.tif','Select first image mask file');
        disp('--- Please load next image mask file ---')
        mask_file_name{1,2} = uigetfile('*.tif','Select next image mask file');
        prompt = 'Do you want to load more deformed image mask files? (0-Yes; 1-No)';
        DoYouWantToLoadMoreImages = input(prompt); imageNo = 2;
        while ( DoYouWantToLoadMoreImages == 0 )   
            imageNo = imageNo + 1;
            mask_file_name{1,imageNo} = uigetfile('*.tif','Select next image mask file');
            prompt = 'Do you want to load more image mask files? (0-Yes; 1-No)';
            DoYouWantToLoadMoreImages = input(prompt);
        end
end

% ==============================================
% The following codes only consider two images comparasion
numImages = size(mask_file_name,2);
for i = 1:numImages
    ImgMask{i} = imread(mask_file_name{1,i});
    % Change color RGB images to grayscale images
    [~, ~, numberOfColorChannels] = size(ImgMask{i});
    if (numberOfColorChannels==3)
        ImgMask{i} = rgb2gray(ImgMask{i});
    end
    ImgMask{i} = logical((ImgMask{i}))';  % Consider the image coordinates
end

% ====== COMMENT ======
% Images physical world coordinates and image coordinates are different:
% --------------------
% --  This is image --
% |                  |
% y                  |
% |                  |
% |  --> x direction |
% |                  |
% --------------------
% after transforming,  MatLab matrix direction:
% --  This is matrix in Matlab --
% |                             |
% x                             |
% |                             |
% |  --> y direction            |
% |                             |
% --------------------------------
 
 
end