function DICmesh = mesh_setup(x,y,DICpara)
%FUNCTION DICmesh = mesh_setup(x,y,DICpara)
% Objective: To set up a DIC uniform FE-mesh
% ----------------------------------------------
%
%   INPUT: x,y      DIC subsets positions
%          DICpara  DIC parameters
%
%   OUTPUT: DICmesh Generated DIC FE-mesh {coordinatesFEM, elementsFEM, ...}
%
% ----------------------------------------------
% Author: Jin Yang.
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================


%% Initialization
winstepsize = DICpara.winstepsize;
ImgSize = DICpara.ImgSize;

meshForLocalMethodOrNot = 0;

%% mesh for global method

% Zach modified
M = size(x,2);  N = size(x,1);   % N is vertically in image; M is horizontally in image;
x = x'; y = y';
coordinatesFEM = [x(:),y(:)];
col1 = repmat((1:(M-1))', N-1, 1) + kron((0:(N-2))'*(M), ones(M-1, 1));
col2 = col1 + 1;
col3 = col1 + M + 1;
col4 = col1 + M;
elementsFEM = [col1, col2, col3, col4];


% I have transpose x and y because Matlab is read matrix in column direction
% for i = 1:size(coordinatesFEM,1)
%     coordinatesFEM(i,:)  = [x(i),y(i)];
%     % x is horizontal position in the image
%     % y is vertical position in the image
% end

% elementsFEM = zeros((M-1)*(N-1),4);
% for j = 1:N-1
%     for i = 1:M-1
%         elementsFEM1((j-1)*(M-1)+i ,:) = [(j-1)*(M)+i (j-1)*(M)+i+1 j*(M)+i+1 j*(M)+i];
%     end
% end


% mesh for local method
% N is vertically in image; M is horizontally in image;

% Zach modified
if meshForLocalMethodOrNot == 1
    gridxtemp = [x(1,1)-0.5*winstepsize; x(:,1)+0.5*winstepsize];
    gridytemp = [y(1,1)-0.5*winstepsize y(1,:)+0.5*winstepsize]';
    [gridx,gridy]=meshgrid(gridxtemp,gridytemp);
    gridx = gridx'; gridy = gridy';
    coordinates = [gridx(:),gridy(:)];
    col11 = repmat((1:M)', N, 1) + kron((0:(N-1))'*(M+1), ones(M, 1));
    col22 = col11 + 1;
    col33 = col11 + M + 2;
    col44 = col11 + M + 1;
    elements = [col11, col22, col33, col44];
end

% elements = zeros( M * N ,4);
% for j = 1:N
%     for i = 1:M
%         elements((j-1)*(M)+i ,:) = [(j-1)*(M+1)+i  (j-1)*(M+1)+i+1  j*(M+1)+i+1   j*(M+1)+i];
%     end
% end

% ======== Assign BC values ==========
% -------- dirichlet BC --------
% dirichlet = [linspace(1,(M-1),(M-1))', linspace(2,M,(M-1))' ;
%             linspace(1,1+M*(N-2),(N-1))', linspace((1+M),(1+M*(N-1)),(N-1))' ;
%             linspace(M,M*(N-1),(N-1))', linspace(2*M,M*N,(N-1))' ;
%             linspace((M*(N-1)+1), (M*N-1), (M-1))', linspace((M*(N-1)+2), (M*N), (M-1))' ];
dirichlet = [];
% -------- neumann BC --------
% neumann = [linspace(1,(M-1),(M-1))', linspace(2,M,(M-1))', zeros(M-1,1), -ones(M-1,1) ;
%              linspace(1,1+M*(N-2),(N-1))', linspace((1+M),(1+M*(N-1)),(N-1))', -ones(N-1,1), zeros(N-1,1) ;
%              linspace(M,M*(N-1),(N-1))', linspace(2*M,M*N,(N-1))', ones(N-1,1), zeros(N-1,1) ;
%              linspace((M*(N-1)+1), (M*N-1), (M-1))', linspace((M*(N-1)+2), (M*N), (M-1))', zeros(M-1,1), ones(M-1,1) ];
neumann = [];



%% Assign variables
DICmesh.coordinatesFEM = coordinatesFEM;
DICmesh.elementsFEM = elementsFEM;
if meshForLocalMethodOrNot == 1
    DICmesh.coordinates = coordinates;
    DICmesh.elements = elements;
end
DICmesh.dirichlet = dirichlet;
DICmesh.neumann = neumann;
DICmesh.x0 = x; DICmesh.y0 = y; DICmesh.M = M; DICmesh.N = N;

DICmesh.y0World = (ImgSize(2)+1-DICmesh.y0);
DICmesh.coordinatesFEMWorld = [DICmesh.coordinatesFEM(:,1),ImgSize(2)+1-DICmesh.coordinatesFEM(:,2)];



