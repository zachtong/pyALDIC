function U = smooth_disp_rbf(U,DICmesh,DICpara)
%SMOOTH_DISP_RBF  Smooth displacement fields by sparse Gaussian kernel.
%
%   U = smooth_disp_rbf(U, DICmesh, DICpara)
%
%   INPUT:  U        - Displacement vector [u1,v1,...,uN,vN]'
%           DICmesh  - DIC mesh structure
%           DICpara  - DIC parameters (.DispSmoothness, .ImgRefMask, etc.)
%
%   OUTPUT: U        - Smoothed displacement field
%
% Uses sparse Gaussian kernel smoothing (O(N log N)) instead of global RBF.

coordinatesFEM = DICmesh.coordinatesFEM;
U = full(U);
smoothness = DICpara.DispSmoothness;

%% Sparse Gaussian smoothing
U = smooth_field_sparse(U, coordinatesFEM, DICpara, 2, smoothness);

%% Fill NaN values
nanindex = find(isnan(U(1:2:end))==1);
notnanindex = setdiff(1:size(coordinatesFEM,1), nanindex);

if ~isempty(nanindex) && ~isempty(notnanindex)
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex-1),'nearest','nearest');
    U1 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));
    Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),U(2*notnanindex),'nearest','nearest');
    U2 = Ftemp(coordinatesFEM(:,1),coordinatesFEM(:,2));

    U = [U1(:),U2(:)]'; U = U(:);
end

end
