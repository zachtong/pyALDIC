function F = smooth_strain_rbf(F,DICmesh,DICpara)
%SMOOTH_STRAIN_RBF  Smooth strain/deformation gradient fields by sparse Gaussian kernel.
%
%   F = smooth_strain_rbf(F, DICmesh, DICpara)
%
%   INPUT:  F        - Deformation gradient [F11,F21,F12,F22,...,F11_N,F21_N,F12_N,F22_N]'
%           DICmesh  - DIC mesh structure
%           DICpara  - DIC parameters (.StrainSmoothness, .ImgRefMask, etc.)
%
%   OUTPUT: F        - Smoothed strain field
%
% Uses sparse Gaussian kernel smoothing (O(N log N)) instead of global RBF.

coordinatesFEM = DICmesh.coordinatesFEM;
F = full(F);
smoothness = DICpara.StrainSmoothness;

%% Sparse Gaussian smoothing
F = smooth_field_sparse(F, coordinatesFEM, DICpara, 4, smoothness);

%% Fill NaN values
nanindexF = find(isnan(F(1:4:end))==1);
notnanindexF = setdiff(1:size(coordinatesFEM,1), nanindexF);

if ~isempty(nanindexF) && ~isempty(notnanindexF)
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

end
