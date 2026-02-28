function [file_name, Img, DICpara] = read_images(varargin)
%read_images  Load DIC images and gather basic DIC parameters.
%
%   [file_name, Img, DICpara] = read_images()
%       Fully interactive: prompts for image files, ROI, subset size, etc.
%
%   [file_name, Img, DICpara] = read_images(DICpara)
%       Skips prompts for any DICpara fields that are already set
%       (winsize, winstepsize, gridxyROIRange, imgBitDepth, LoadImgMethod).
%       Still prompts for image file selection.
%
%   [file_name, Img, DICpara] = ReadImageQuadtree(DICpara, file_name, Img)
%       Fully programmatic: skips all interactive prompts. Useful for
%       tests, batch processing, and GUI integration.
%
%   INPUTS (all optional):
%       DICpara   - struct with pre-set parameter fields
%       file_name - cell array of image file names {1 x nFrames}
%       Img       - cell array of loaded images (double, transposed)
%
%   OUTPUTS:
%       file_name - cell array of image file names
%       Img       - cell array of loaded images (double, transposed)
%       DICpara   - struct with fields set by this function:
%                   .imgBitDepth, .winsize, .winstepsize,
%                   .gridxyROIRange, .LoadImgMethod, .ImgSize
%                   All other fields are filled by DICpara_default().
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
        file_name = varargin{2};
    else
        file_name = {};
    end
    if nargin >= 3 && ~isempty(varargin{3})
        Img = varargin{3};
    else
        Img = {};
    end

    %% ====== Section 1: Load images ======
    if isempty(file_name)
        % --- Interactive file selection ---
        if needsInput(DICpara, 'LoadImgMethod')
            fprintf('Choose method to load images:  \n')
            fprintf('     0: Select images folder;  \n')
            fprintf('     1: Use prefix of image names;  \n')
            fprintf('     2: Manually select images.  \n')
            prompt = 'Input here: ';
            LoadImgMethod = input(prompt);
        else
            LoadImgMethod = DICpara.LoadImgMethod;
        end

        switch LoadImgMethod
            case 0
                imgfoldername = uigetdir(pwd, 'Select images folder');
                addpath([imgfoldername, '\']);
                img1 = dir(fullfile(imgfoldername, '*.jpg'));
                img2 = dir(fullfile(imgfoldername, '*.jpeg'));
                img3 = dir(fullfile(imgfoldername, '*.tif'));
                img4 = dir(fullfile(imgfoldername, '*.tiff'));
                img5 = dir(fullfile(imgfoldername, '*.bmp'));
                img6 = dir(fullfile(imgfoldername, '*.png'));
                img7 = dir(fullfile(imgfoldername, '*.jp2'));
                file_name = [img1; img2; img3; img4; img5; img6; img7];
                file_name = struct2cell(file_name);
            case 1
                fprintf('What is prefix of DIC images? E.g. img_0*.tif.   \n')
                prompt = 'Input here: ';
                file_name = input(prompt, 's');
                [~, imgname, imgext] = fileparts(file_name);
                file_name = dir([imgname, imgext]);
                file_name = struct2cell(file_name);
            otherwise
                disp('--- Please load first image ---')
                file_name{1,1} = uigetfile('*.tif', 'Select reference Image (Deformed)');
                disp('--- Please load next image ---')
                file_name{1,2} = uigetfile('*.tif', 'Select deformed Image (Reference)');
                prompt = 'Do you want to load more deformed images? (0-Yes; 1-No)';
                DoYouWantToLoadMoreImages = input(prompt);
                imageNo = 2;
                while (DoYouWantToLoadMoreImages == 0)
                    imageNo = imageNo + 1;
                    file_name{1,imageNo} = uigetfile('*.tif', 'Select Deformed Image');
                    prompt = 'Do you want to load more deformed images? (0-Yes; 1-No)';
                    DoYouWantToLoadMoreImages = input(prompt);
                end
        end

        DICpara.LoadImgMethod = LoadImgMethod;
    end

    if isempty(Img)
        % --- Read images from disk ---
        % Images physical world coordinates and image coordinates are different:
        % Image coords: x = horizontal (columns), y = vertical (rows)
        % After transposing: MATLAB matrix x = rows, y = columns
        numImages = size(file_name, 2);
        for i = 1:numImages
            Img{i} = imread(file_name{1,i});
            [~, ~, numberOfColorChannels] = size(Img{i});
            if numberOfColorChannels == 3
                Img{i} = rgb2gray(Img{i});
            elseif numberOfColorChannels == 4
                Img{i} = rgb2gray(Img{i}(:,:,1:3));
            end
            Img{i} = double(Img{i})';
        end
    end

    %% ====== Section 2: Image bit depth ======
    if needsInput(DICpara, 'imgBitDepth')
        if ~isempty(file_name)
            imgInfo = imfinfo(file_name{1});
            DICpara.imgBitDepth = imgInfo.BitDepth;
        else
            DICpara.imgBitDepth = 8; % Fallback for synthetic images
        end
    end

    %% ====== Section 3: ROI selection ======
    if needsROI(DICpara) && ~isempty(file_name)
        % Interactive ROI selection via ginput
        fprintf('\n');
        disp('--- Please change the value of "DICpara.imgBitDepth" if your image is too dark/bright ---');
        disp('--- Define ROI corner points at the top-left and the bottom-right ---')

        imgForROI = imread(file_name{1});
        [~, ~, numChannels] = size(imgForROI);

        if numChannels == 1
            imshow(imgForROI, [0, 2^DICpara.imgBitDepth - 1]);
        elseif numChannels == 3
            imshow(imgForROI);
        elseif numChannels == 4
            imshow(imgForROI(:,:,1:3));
        end

        title('Click top-left and the bottom-right corner points', ...
              'fontweight', 'normal', 'fontsize', 16);

        gridx = zeros(1,2); gridy = zeros(1,2);
        [gridx(1), gridy(1)] = ginput(1);
        fprintf('Coordinates of top-left corner point are (%4.3f,%4.3f)\n', gridx(1), gridy(1))

        [gridx(2), gridy(2)] = ginput(1);
        fprintf('Coordinates of bottom-right corner point are (%4.3f,%4.3f)\n', gridx(2), gridy(2))

        DICpara.gridxyROIRange.gridx = round(gridx);
        DICpara.gridxyROIRange.gridy = round(gridy);
    end

    %% ====== Section 4: Subset size ======
    if needsInput(DICpara, 'winsize')
        fprintf('\n');
        fprintf('--- What is the subset size? --- \n');
        fprintf('Each subset has an area of [-winsize/2:winsize/2, -winsize/2:winsize/2] \n');
        prompt = 'Input an even number (E.g., 32): ';
        DICpara.winsize = input(prompt);
    end

    %% ====== Section 5: Subset step ======
    if needsInput(DICpara, 'winstepsize')
        fprintf('--- What is the subset step? --- \n');
        prompt = 'Input an integer to be a power of 2 (E.g., 16):  ';
        DICpara.winstepsize = input(prompt);
    end

    %% ====== Section 6: Image size (always set from loaded images) ======
    DICpara.ImgSize = size(Img{1});

end


%% ====== Helper functions ======

function tf = needsInput(S, fieldName)
%NEEDSINPUT  True if the field is missing or empty in struct S.
    tf = ~isfield(S, fieldName) || isempty(S.(fieldName));
end

function tf = needsROI(S)
%NEEDSROI  True if gridxyROIRange.gridx is missing or empty.
    tf = ~isfield(S, 'gridxyROIRange') || ...
         ~isstruct(S.gridxyROIRange) || ...
         ~isfield(S.gridxyROIRange, 'gridx') || ...
         isempty(S.gridxyROIRange.gridx);
end
