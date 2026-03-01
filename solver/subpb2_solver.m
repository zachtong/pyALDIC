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
tripI = zeros(nEle * nnzPerEle, 1);
tripJ = zeros(nEle * nnzPerEle, 1);
for e = 1:nEle
    idx = (e-1)*nnzPerEle + (1:nnzPerEle);
    tripI(idx) = allIndexU(e, localR);
    tripJ(idx) = allIndexU(e, localC);
end

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

    for e = 1:nEle
        Ne = N_all(:,:,e);    % 2 x 16
        DNe = DN_all(:,:,e);  % 4 x 16
        w = weight(e);

        % Stiffness: tempA += w * ((beta+alpha)*DN'*DN + mu*N'*N)
        NtN = Ne' * Ne;        % 16 x 16
        DtD = DNe' * DNe;      % 16 x 16
        tempA_all(e,:) = tempA_all(e,:) + w * ((beta+alpha)*DtD(:)' + mu*NtN(:)');

        % Load: tempb += w * (beta*diag(DN'*FW') + mu*N'*N*Uv + alpha*DN'*DN*U)
        FW_e = squeeze(FMinusW_ele(e,:,:));  % 16 x 4
        be = w * ( beta * sum(DNe' .* FW_e, 2) ...
             + mu * NtN * UMinusv_ele(e,:)' ...
             + alpha * DtD * U_ele(e,:)' );
        bV_all(e,:) = bV_all(e,:) + be';
    end
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
Uhat(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);
Uhat = full(Uhat(1:FEMSize));


end


%% ========= Vectorized shape function computation for all elements ========
function [N_all, DN_all, Jdet_all] = compute_all_elements_gp( ...
        ksi, eta, ptx, pty, delta, nEle)
% Compute shape functions, DN matrix, and Jacobian determinant for all
% elements at one Gauss point (ksi, eta).
%
% Inputs:
%   ksi, eta   - scalar Gauss point in reference coords
%   ptx, pty   - nEle x 8, element node coordinates
%   delta      - nEle x 4, midpoint flags for nodes 5-8
%   nEle       - number of elements
%
% Outputs:
%   N_all    - 2 x 16 x nEle, shape function matrix
%   DN_all   - 4 x 16 x nEle, shape function gradient matrix (physical)
%   Jdet_all - nEle x 1, Jacobian determinants

d5 = delta(:,1); d6 = delta(:,2); d7 = delta(:,3); d8 = delta(:,4);

% --- Shape functions (nEle x 1 each) ---
N5 = d5 .* (0.5*(1+ksi)*(1-abs(eta)));
N6 = d6 .* (0.5*(1+eta)*(1-abs(ksi)));
N7 = d7 .* (0.5*(1-ksi)*(1-abs(eta)));
N8 = d8 .* (0.5*(1-eta)*(1-abs(ksi)));
N1 = (1-ksi)*(1-eta)*0.25 - 0.5*(N7+N8);
N2 = (1+ksi)*(1-eta)*0.25 - 0.5*(N8+N5);
N3 = (1+ksi)*(1+eta)*0.25 - 0.5*(N5+N6);
N4 = (1-ksi)*(1+eta)*0.25 - 0.5*(N6+N7);

% --- Shape function derivatives w.r.t. ksi (nEle x 1 each) ---
seta = sign(-eta); sksi = sign(-ksi);
dN5k = d5 .* (0.5*(1-abs(eta)));
dN6k = d6 .* (0.5*(1+eta)*sksi);
dN7k = d7 .* (-0.5*(1-abs(eta)));
dN8k = d8 .* (0.5*(1-eta)*sksi);
dN1k = -0.25*(1-eta) - 0.5*(dN7k + dN8k);
dN2k =  0.25*(1-eta) - 0.5*(dN8k + dN5k);
dN3k =  0.25*(1+eta) - 0.5*(dN5k + dN6k);
dN4k = -0.25*(1+eta) - 0.5*(dN6k + dN7k);

% --- Shape function derivatives w.r.t. eta (nEle x 1 each) ---
dN5e = d5 .* (0.5*(1+ksi)*seta);
dN6e = d6 .* (0.5*(1-abs(ksi)));
dN7e = d7 .* (0.5*(1-ksi)*seta);
dN8e = d8 .* (-0.5*(1-abs(ksi)));
dN1e = -0.25*(1-ksi) - 0.5*(dN7e + dN8e);
dN2e = -0.25*(1+ksi) - 0.5*(dN8e + dN5e);
dN3e =  0.25*(1+ksi) - 0.5*(dN5e + dN6e);
dN4e =  0.25*(1-ksi) - 0.5*(dN6e + dN7e);

% Stack dN/dksi and dN/deta: nEle x 8
dNdk = [dN1k, dN2k, dN3k, dN4k, dN5k, dN6k, dN7k, dN8k];
dNde = [dN1e, dN2e, dN3e, dN4e, dN5e, dN6e, dN7e, dN8e];

% --- Jacobian: J = [J11 J12; J21 J22] per element ---
J11 = sum(dNdk .* ptx, 2);
J12 = sum(dNdk .* pty, 2);
J21 = sum(dNde .* ptx, 2);
J22 = sum(dNde .* pty, 2);
Jdet_all = J11.*J22 - J12.*J21;

% --- InvJ * [dN/dksi; dN/deta] -> [dN/dx; dN/dy] per element ---
invDet = 1 ./ Jdet_all;
dNdx = invDet .* ( J22 .* dNdk - J12 .* dNde);
dNdy = invDet .* (-J21 .* dNdk + J11 .* dNde);

% --- Build N_all (2 x 16 x nEle) ---
Nvals = [N1, N2, N3, N4, N5, N6, N7, N8];
N_all = zeros(2, 16, nEle);
for k = 1:8
    N_all(1, 2*k-1, :) = Nvals(:, k);
    N_all(2, 2*k,   :) = Nvals(:, k);
end

% --- Build DN_all (4 x 16 x nEle) ---
DN_all = zeros(4, 16, nEle);
for k = 1:8
    DN_all(1, 2*k-1, :) = dNdx(:, k);
    DN_all(2, 2*k-1, :) = dNdy(:, k);
    DN_all(3, 2*k,   :) = dNdx(:, k);
    DN_all(4, 2*k,   :) = dNdy(:, k);
end

end
