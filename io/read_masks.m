function [mask_file_name, ImgMask] = read_masks(varargin)
%read_masks  Load DIC mask images.
%
%   [mask_file_name, ImgMask] = read_masks()
%       Fully interactive: prompts for mask files.
%
%   [mask_file_name, ImgMask] = read_masks(DICpara)
%       Skips prompts for fields already set in DICpara
%       (LoadMaskMethod, maskFolder).
%
%   [mask_file_name, ImgMask] = read_masks(DICpara, mask_file_name, ImgMask)
%       Fully programmatic: skips all interactive prompts.
%
%   INPUTS (all optional):
%       DICpara        - struct with pre-set parameter fields
%       mask_file_name - cell array of mask file names {1 x nFrames}
%       ImgMask        - cell array of loaded mask images (logical, transposed)
%
%   OUTPUTS:
%       mask_file_name - cell array of mask file names
%       ImgMask        - cell array of loaded mask images (logical, transposed)
%
% ----------------------------------------------
% Author: Jin Yang.
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2026. Refactored for programmatic usage.
% ==============================================

    %% Parse optional inputs
    if nargin >= 1 && isstruct(varargin{1})
        DICpara = varargin{1};
    else
        DICpara = struct();
    end
    if nargin >= 2 && ~isempty(varargin{2})
        mask_file_name = varargin{2};
    else
        mask_file_name = {};
    end
    if nargin >= 3 && ~isempty(varargin{3})
        ImgMask = varargin{3};
    else
        ImgMask = {};
    end

    %% Section 1: Load mask file names
    if isempty(mask_file_name)
        if needsInput(DICpara, 'LoadMaskMethod')
            fprintf('Choose method to load image mask files:  \n')
            fprintf('     0: Select images mask file folder;  \n')
            fprintf('     1: Use prefix of image mask file names;  \n')
            fprintf('     2: Manually select image mask files.  \n')
            prompt = 'Input here: ';
            LoadMaskMethod = input(prompt);
        else
            LoadMaskMethod = DICpara.LoadMaskMethod;
        end

        switch LoadMaskMethod
            case 0
                if needsInput(DICpara, 'maskFolder')
                    imgfoldername = uigetdir(pwd, 'Select image mask file folder');
                else
                    imgfoldername = DICpara.maskFolder;
                end
                addpath(imgfoldername);
                img1 = dir(fullfile(imgfoldername, '*.jpg'));
                img2 = dir(fullfile(imgfoldername, '*.jpeg'));
                img3 = dir(fullfile(imgfoldername, '*.tif'));
                img4 = dir(fullfile(imgfoldername, '*.tiff'));
                img5 = dir(fullfile(imgfoldername, '*.bmp'));
                img6 = dir(fullfile(imgfoldername, '*.png'));
                img7 = dir(fullfile(imgfoldername, '*.jp2'));
                mask_file_name = [img1; img2; img3; img4; img5; img6; img7];
                mask_file_name = struct2cell(mask_file_name);
            case 1
                fprintf('What is prefix of DIC images? E.g. img_0*.tif.   \n')
                prompt = 'Input here: ';
                mask_file_name = input(prompt, 's');
                [~, imgname, imgext] = fileparts(mask_file_name);
                mask_file_name = dir([imgname, imgext]);
                mask_file_name = struct2cell(mask_file_name);
            otherwise
                disp('--- Please load first image mask file ---')
                mask_file_name{1,1} = uigetfile('*.tif', 'Select first image mask file');
                disp('--- Please load next image mask file ---')
                mask_file_name{1,2} = uigetfile('*.tif', 'Select next image mask file');
                prompt = 'Do you want to load more deformed image mask files? (0-Yes; 1-No)';
                DoYouWantToLoadMoreImages = input(prompt);
                imageNo = 2;
                while (DoYouWantToLoadMoreImages == 0)
                    imageNo = imageNo + 1;
                    mask_file_name{1,imageNo} = uigetfile('*.tif', 'Select next image mask file');
                    prompt = 'Do you want to load more image mask files? (0-Yes; 1-No)';
                    DoYouWantToLoadMoreImages = input(prompt);
                end
        end
    end

    %% Section 2: Load mask images from disk
    if isempty(ImgMask)
        numImages = size(mask_file_name, 2);
        for i = 1:numImages
            ImgMask{i} = imread(mask_file_name{1,i});
            [~, ~, numberOfColorChannels] = size(ImgMask{i});
            if (numberOfColorChannels == 3)
                ImgMask{i} = rgb2gray(ImgMask{i});
            end
            ImgMask{i} = logical((ImgMask{i}))';  % Consider the image coordinates
        end
    end

end


%% ====== Helper functions ======

function tf = needsInput(S, fieldName)
%NEEDSINPUT  True if the field is missing or empty in struct S.
    tf = ~isfield(S, fieldName) || isempty(S.(fieldName));
end
