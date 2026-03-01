function V = fill_nan_rbf(V, coordinatesFEM, imgSize, ImgRefMask, nComponents)
%FILL_NAN_RBF  Fill NaN values in displacement/gradient vectors via scatteredInterpolant.
%
%   V = fill_nan_rbf(V, coordinatesFEM, imgSize, ImgRefMask, nComponents)
%
%   Uses scatteredInterpolant with 'natural' method for NaN filling.
%   O(N log N) complexity.
%
%   INPUTS:
%     V              - column vector: interleaved components (2 for disp, 4 for F)
%     coordinatesFEM - nNodes x 2 coordinates
%     imgSize        - [rows, cols] image size (unused, kept for API compat)
%     ImgRefMask     - logical mask image (unused, kept for API compat)
%     nComponents    - 2 (displacement only) or 4 (displacement + deformation gradient)
%
%   OUTPUT:
%     V              - NaN-filled vector

    nNodes = size(coordinatesFEM, 1);
    nc = nComponents;

    % Identify NaN nodes
    nanindex = find(isnan(V(1:nc:end)));
    notnanindex = setdiff(1:nNodes, nanindex);
    if isempty(nanindex), return; end

    if ~isempty(notnanindex)
        for c = 1:nc
            Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), ...
                V(nc*notnanindex - (nc-c)), 'natural', 'nearest');
            vals = Ftemp(coordinatesFEM(:,1), coordinatesFEM(:,2));
            V(c:nc:end) = vals(:);
        end
    else
        warning('fill_nan_rbf:allNaN', 'All nodes are NaN, cannot interpolate.');
    end
end
