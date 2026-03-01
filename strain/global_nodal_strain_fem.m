function [StrainNodalPt] = global_nodal_strain_fem(DICmesh, DICpara, U)
%GLOBAL_NODAL_STRAIN_FEM  Compute deformation gradient via FEM shape function derivatives.
%
%   [StrainNodalPt] = global_nodal_strain_fem(DICmesh, DICpara, U)
%
%   Evaluates displacement gradients (du/dx, du/dy, dv/dx, dv/dy) at each
%   element center using FEM shape function derivatives, then averages to
%   nodes weighted by element area. O(nElements) complexity.
%
%   INPUTS:
%     DICmesh  - struct with .coordinatesFEM (nNode x 2), .elementsFEM (nEle x 8)
%     DICpara  - struct with .winstepsize, .ImgRefMask, .ImgSize
%     U        - displacement vector [2*nNode x 1]: [u1,v1,...,uN,vN]'
%
%   OUTPUT:
%     StrainNodalPt - deformation gradient [4*nNode x 1]:
%                     [F11,F21,F12,F22,...] = [du/dx,dv/dx,du/dy,dv/dy,...]

coordinatesFEM = DICmesh.coordinatesFEM;
elementsFEM = DICmesh.elementsFEM;
winstepsize = DICpara.winstepsize;
nNodes = size(coordinatesFEM, 1);
nEle = size(elementsFEM, 1);

%% Gather element node coordinates
ptx = zeros(nEle, 8);
pty = zeros(nEle, 8);
for k = 1:8
    valid = elementsFEM(:,k) > 0;
    ptx(valid, k) = coordinatesFEM(elementsFEM(valid,k), 1);
    pty(valid, k) = coordinatesFEM(elementsFEM(valid,k), 2);
end
delta = double(elementsFEM(:,5:8) ~= 0);  % hanging-node flags

%% Gather element displacements
dummyNode = nNodes + 1;
allIndexU = zeros(nEle, 16);
for k = 1:8
    nodeIds = elementsFEM(:, k);
    nodeIds(nodeIds == 0) = dummyNode;
    allIndexU(:, 2*k-1) = 2*nodeIds - 1;
    allIndexU(:, 2*k)   = 2*nodeIds;
end
U_ext = [U(:); 0; 0];  % pad for dummy node
U_ele = U_ext(allIndexU);  % nEle x 16

%% Evaluate strain at element center (ksi=0, eta=0)
[~, DN_all, Jdet] = compute_all_elements_gp(0, 0, ptx, pty, delta, nEle);

% Extract dN/dx and dN/dy for the 8 shape functions
% DN_all is 4 x 16 x nEle; rows: [du/dx, du/dy, dv/dx, dv/dy]
% Odd columns (1,3,5,...,15) correspond to u-DOFs → rows 1,2 give dN/dx, dN/dy
dNdx = squeeze(DN_all(1, 1:2:16, :))';  % nEle x 8
dNdy = squeeze(DN_all(2, 1:2:16, :))';  % nEle x 8

% Element u and v components
u_ele = U_ele(:, 1:2:16);  % nEle x 8
v_ele = U_ele(:, 2:2:16);  % nEle x 8

% Displacement gradients at element center
dudx_ele = sum(dNdx .* u_ele, 2);  % F11: du/dx
dudy_ele = sum(dNdy .* u_ele, 2);  % F12: du/dy
dvdx_ele = sum(dNdx .* v_ele, 2);  % F21: dv/dx
dvdy_ele = sum(dNdy .* v_ele, 2);  % F22: dv/dy

%% Area-weighted averaging to nodes
eleArea = abs(Jdet);  % element area proxy (Jacobian det at center * 4)
F_nodal = zeros(nNodes, 4);   % columns: [F11, F21, F12, F22]
nodeWeight = zeros(nNodes, 1);

for k = 1:8
    nodeIds = elementsFEM(:, k);
    valid = nodeIds > 0;
    nids = nodeIds(valid);
    w = eleArea(valid);

    % Accumulate weighted strain contributions
    F_nodal(nids, 1) = F_nodal(nids, 1) + w .* dudx_ele(valid);
    F_nodal(nids, 2) = F_nodal(nids, 2) + w .* dvdx_ele(valid);
    F_nodal(nids, 3) = F_nodal(nids, 3) + w .* dudy_ele(valid);
    F_nodal(nids, 4) = F_nodal(nids, 4) + w .* dvdy_ele(valid);
    nodeWeight(nids) = nodeWeight(nids) + w;
end

% Divide by total weight; orphan nodes (no elements) get NaN
orphan = nodeWeight < eps;
nodeWeight(orphan) = 1;  % avoid division by zero
F_nodal = F_nodal ./ nodeWeight;
F_nodal(orphan, :) = NaN;

%% Interleave to output format: [F11,F21,F12,F22] per node
StrainNodalPt = reshape(F_nodal', [], 1);  % 4*nNodes x 1

%% Outlier removal (same as global_nodal_strain_rbf.m)
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

%% Fill NaN values via scatteredInterpolant
nanindexF = find(isnan(StrainNodalPt(1:4:end))==1);
notnanindexF = setdiff(1:nNodes, nanindexF);

if ~isempty(nanindexF) && ~isempty(notnanindexF)
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-3),'linear','linear');
    F11 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-2),'linear','linear');
    F21 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-1),'linear','linear');
    F12 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),StrainNodalPt(4*notnanindexF-0),'linear','linear');
    F22 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

    StrainNodalPt = [F11(:),F21(:),F12(:),F22(:)]';
    StrainNodalPt = StrainNodalPt(:);
end

end
