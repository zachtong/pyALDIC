function [Uhat] = subpb2_solver(DICmesh,GaussPtOrder,beta,mu,U,F,udual,vdual, ...
                                 alpha,winstepsize)
%FUNCTION [Uhat] = subpb2_solver(DICmesh,GaussPtOrder,beta,mu, ...
%          U,F,udual,vdual,alpha,winstepsize)
% AL-DIC Subproblem 2 is solved over a quadtree mesh to find a globally
% kinematically compatible deformation field by finite element method.
% Uses vectorized FEM assembly over all elements simultaneously.
% ----------------------------------------------
%
%   INPUT: DICmesh             DIC FE Q4 mesh: coordinatesFEM, elementsFEM
%          GaussPtOrder        Gauss point order used in FE Q4 element
%          beta, mu            Two constant coefficients
%          U                   Disp vector: U = [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%          F                   Deformation gradient: F = [F11_node1, F21_node1, F12_node1, F22_node1, ...
%                                                         ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%          udual,vdual         Dual variables
%          alpha               Smoothness coefficient. Not needed here, i.e., alpha=0
%          winstepsize         DIC FE Q4 mesh spacing
%
%   OUTPUT: Uhat               Solved globally kinematically compatible displacement field
%
% ----------------------------------------------
% Author: Jin Yang.
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 2018.03, 2020.12
% ==============================================


%% Initialization
coordinatesFEM = DICmesh.coordinatesFEM;
elementsFEM = DICmesh.elementsFEM;
dirichlet = DICmesh.dirichlet;
neumann = DICmesh.neumann;

DIM = 2;
NodesNumPerEle = 4;
FEMSize = DIM*size(coordinatesFEM,1);


%% ====== Initialize variables ======
U = [U;zeros(DIM*NodesNumPerEle,1)]; v = [0*vdual;zeros(DIM*NodesNumPerEle,1)];
F = [F;zeros(DIM^2*NodesNumPerEle,1)]; W = [0*udual;zeros(DIM^2*NodesNumPerEle,1)];
UMinusv = U-v; FMinusW = F-W;

% ====== Gaussian quadrature parameter ======
switch GaussPtOrder
    case 2 % 2*2 Gauss points
        gqpt1 = -1/sqrt(3); gqpt2 = 1/sqrt(3); gqpt = [gqpt1,gqpt2];
        gqwt1 = 1; gqwt2 = 1; gqwt = [gqwt1,gqwt2];
    case 3 % 3*3 Gauss points
        gqpt1 = 0; gqpt2 = sqrt(3/5); gqpt3 = -sqrt(3/5); gqpt = [gqpt1,gqpt2,gqpt3];
        gqwt1 = 8/9; gqwt2 = 5/9; gqwt3 = 5/9; gqwt = [gqwt1,gqwt2,gqwt3];
    case 4 % 4*4 Gauss points
        gqpt1 = 0.339981; gqpt2 = -0.339981; gqpt3 = 0.861136; gqpt4 = -0.861136;
        gqwt1 = 0.652145; gqwt2 = 0.652145; gqwt3 = 0.347855; gqwt4 = 0.347855;
        gqpt = [gqpt1,gqpt2,gqpt3,gqpt4]; gqwt = [gqwt1,gqwt2,gqwt3,gqwt4];
    case 5 % 5*5 Gauss points
        gqpt1 = 0; gqpt2 = 0.538469; gqpt3 = -0.538469; gqpt4 = 0.90618; gqpt5 = -0.90618;
        gqwt1 = 0.568889; gqwt2 = 0.478629; gqwt3 = 0.478629; gqwt4 = 0.236927; gqwt5 = 0.236927;
        gqpt = [gqpt1,gqpt2,gqpt3,gqpt4,gqpt5]; gqwt = [gqwt1,gqwt2,gqwt3,gqwt4,gqwt5];
    otherwise
        error('subpb2_solver:badGaussPtOrder', 'GaussPtOrder %d not implemented.', GaussPtOrder);
end


%% ====== Vectorized FEM Assembly ======
nEle = size(elementsFEM, 1);
dummyNode = FEMSize/DIM + 1;  % dummy node index for missing midpoints
bigN = FEMSize + DIM*NodesNumPerEle;

% Element node coordinates: nEle x 8
ptx = zeros(nEle, 8); pty = zeros(nEle, 8);
for k = 1:8
    valid = elementsFEM(:,k) > 0;
    ptx(valid, k) = coordinatesFEM(elementsFEM(valid,k), 1);
    pty(valid, k) = coordinatesFEM(elementsFEM(valid,k), 2);
end

% Delta flags for midpoints 5-8
delta = double(elementsFEM(:,5:8) ~= 0);  % nEle x 4

% Build DOF index arrays: allIndexU (nEle x 16), allIndexF (nEle x 16 x 4)
allIndexU = zeros(nEle, 16);
allIndexF = zeros(nEle, 16, 4);
for k = 1:8
    nodeIds = elementsFEM(:, k);
    nodeIds(nodeIds == 0) = dummyNode;
    allIndexU(:, 2*k-1) = 2*nodeIds - 1;
    allIndexU(:, 2*k)   = 2*nodeIds;
    allIndexF(:, 2*k-1, :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
    allIndexF(:, 2*k,   :) = [4*nodeIds-3, 4*nodeIds-1, 4*nodeIds-2, 4*nodeIds];
end

% Gauss points grid
[ksiGrid, etaGrid] = ndgrid(gqpt, gqpt);
[wksiGrid, wetaGrid] = ndgrid(gqwt, gqwt);
ksiList = ksiGrid(:); etaList = etaGrid(:);
wksiList = wksiGrid(:); wetaList = wetaGrid(:);
nGP = length(ksiList);

% Build COO row/col index arrays for sparse A (same structure for all GPs)
nnzPerEle = 256;  % 16 x 16
[localR, localC] = ndgrid(1:16, 1:16);
localR = localR(:)'; localC = localC(:)';  % 1 x 256
tripI = reshape(allIndexU(:, localR)', [], 1);  % nEle*256 x 1
tripJ = reshape(allIndexU(:, localC)', [], 1);  % nEle*256 x 1

% Gather element-level vectors from global vectors
UMinusv_ele = UMinusv(allIndexU);  % nEle x 16
U_ele = U(allIndexU);              % nEle x 16
FMinusW_ele = zeros(nEle, 16, 4);
for c = 1:4
    FMinusW_ele(:,:,c) = FMinusW(allIndexF(:,:,c));
end

% ====== Accumulate stiffness and load over Gauss points ======
tempA_all = zeros(nEle, 256);  % flattened 16x16 per element
bV_all = zeros(nEle, 16);     % load vector per element

for gp = 1:nGP
    ksi = ksiList(gp); eta = etaList(gp);
    wk = wksiList(gp);  we = wetaList(gp);

    % Compute shape functions, DN, Jacobian for ALL elements at once
    [N_all, DN_all, Jdet] = compute_all_elements_gp(ksi, eta, ptx, pty, delta, nEle);

    weight = Jdet * (wk * we);  % nEle x 1

    % Batch matrix products via pagemtimes (R2020b+): 16x16xnEle
    NtN = pagemtimes(permute(N_all,[2,1,3]), N_all);    % 16x16xnEle
    DtD = pagemtimes(permute(DN_all,[2,1,3]), DN_all);   % 16x16xnEle

    % Stiffness: reshape to nEle x 256, scale by weight
    NtN_flat = reshape(NtN, 256, nEle)';   % nEle x 256
    DtD_flat = reshape(DtD, 256, nEle)';   % nEle x 256
    tempA_all = tempA_all + weight .* ((beta+alpha)*DtD_flat + mu*NtN_flat);

    % Load vector: three terms
    % Term 1: beta * sum(DN' .* FW, 2)  per element
    DNt = permute(DN_all, [2,1,3]);              % 16x4xnEle
    FW_paged = permute(FMinusW_ele, [2,3,1]);    % 16x4xnEle
    term1 = squeeze(sum(DNt .* FW_paged, 2));    % 16xnEle

    % Term 2: mu * N'*N * UMinusv
    Uv_paged = reshape(UMinusv_ele', 16, 1, nEle);  % 16x1xnEle
    term2 = reshape(pagemtimes(NtN, Uv_paged), 16, nEle);

    % Term 3: alpha * DN'*DN * U
    Ue_paged = reshape(U_ele', 16, 1, nEle);  % 16x1xnEle
    term3 = reshape(pagemtimes(DtD, Ue_paged), 16, nEle);

    be = weight' .* (beta * term1 + mu * term2 + alpha * term3);  % 16 x nEle
    bV_all = bV_all + be';
end

% ====== Sparse assembly (single call each) ======
tripV = reshape(tempA_all', [], 1);
A = sparse(tripI, tripJ, tripV, bigN, bigN);

bI_all = allIndexU';   % 16 x nEle
bV_col = bV_all';      % 16 x nEle
b = sparse(bI_all(:), ones(numel(bI_all),1), bV_col(:), bigN, 1);


%% ====== Boundary conditions and solve ======
% Finding evolved nodal points
coordsIndexInvolved = unique(elementsFEM(:,1:8));
UIndexInvolved = zeros(2*(length(coordsIndexInvolved)-1),1);
for tempi = 1:(size(coordsIndexInvolved,1)-1)
    UIndexInvolved(2*tempi-1:2*tempi) = [2*coordsIndexInvolved(tempi+1)-1; 2*coordsIndexInvolved(tempi+1)];
end

% Set Dirichlet and Neumann boundary conditions
if ~isempty(dirichlet)
    dirichlettemp = [2*dirichlet(:); 2*dirichlet(:)-1];
else
    dirichlettemp = [];
end
FreeNodes = setdiff(UIndexInvolved, unique(dirichlettemp));

% Neumann conditions
BCForce = - 1/winstepsize * F;
for tempj = 1:size(neumann,1)
    b(2*neumann(tempj,1:2)-1) = b(2*neumann(tempj,1:2)-1) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
      *( ( BCForce(4*neumann(tempj,1:2)-3) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)-1) * neumann(tempj,4) ) );
    b(2*neumann(tempj,1:2))   = b(2*neumann(tempj,1:2)) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
     * (( BCForce(4*neumann(tempj,1:2)-2) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)) * neumann(tempj,4) ) );
end

% Dirichlet conditions
Uhat = sparse(bigN, 1);
for tempi = 1:DIM
    Uhat(DIM*unique(dirichlet)-(tempi-1)) = U(DIM*unique(dirichlet)-(tempi-1));
end
b = b - A * Uhat;

% Solve FEM problem
Afree = A(FreeNodes, FreeNodes);
bfree = b(FreeNodes);
nFree = length(FreeNodes);
if nFree > 50000
    % Large system: use PCG with incomplete Cholesky preconditioner
    try
        L = ichol(Afree, struct('type','ict','droptol',1e-3));
        [x_pcg, flag] = pcg(Afree, bfree, 1e-6, 1000, L, L');
        if flag ~= 0
            warning('subpb2_solver:pcgNoConverge', ...
                'pcg flag=%d, falling back to direct solver.', flag);
            x_pcg = Afree \ bfree;
        end
    catch
        x_pcg = Afree \ bfree;
    end
    Uhat(FreeNodes) = x_pcg;
else
    Uhat(FreeNodes) = Afree \ bfree;
end
Uhat = full(Uhat(1:FEMSize));


end
