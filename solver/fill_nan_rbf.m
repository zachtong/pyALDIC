function V = fill_nan_rbf(V, coordinatesFEM, imgSize, ImgRefMask, nComponents)
%FILL_NAN_RBF  Fill NaN values in displacement/gradient vectors via RBF interpolation.
%
%   V = fill_nan_rbf(V, coordinatesFEM, imgSize, ImgRefMask, nComponents)
%
%   Uses per-connected-region RBF (thin plate) interpolation, with
%   scatteredInterpolant as fallback for remaining NaNs.
%
%   INPUTS:
%     V              - column vector: interleaved components (2 for disp, 4 for F)
%     coordinatesFEM - nNodes x 2 coordinates
%     imgSize        - [rows, cols] image size for sub2ind
%     ImgRefMask     - logical mask image
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

    % Connected component analysis on mask
    dilatedI = logical(ImgRefMask);
    cc = bwconncomp(dilatedI, 8);
    indPxAll = sub2ind(imgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)));
    indPxNotNanAll = sub2ind(imgSize, round(coordinatesFEM(notnanindex,1)), round(coordinatesFEM(notnanindex,2)));
    stats = regionprops(cc, 'Area', 'PixelList');

    for tempi = 1:length(stats)
        try
            indPxtempi = sub2ind(imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1));
            Lia = ismember(indPxAll, indPxtempi); [LiaList,~] = find(Lia==1);
            Lib = ismember(indPxNotNanAll, indPxtempi); [LibList,~] = find(Lib==1);

            srcCoords = round(coordinatesFEM(notnanindex(LibList), 1:2))';
            dstCoords = coordinatesFEM(LiaList, 1:2)';

            for c = 1:nc
                srcVals = V(nc*notnanindex(LibList) - (nc-c))';
                op1 = rbfcreate(srcCoords, srcVals, 'RBFFunction', 'thinplate');
                fi1 = rbfinterp(dstCoords, op1);
                V(nc*LiaList - (nc-c)) = fi1(:);
            end
        catch ME
            warning('fill_nan_rbf:regionFail', 'RBF NaN fill failed for region %d: %s', tempi, ME.message);
        end
    end

    % Fallback: scatteredInterpolant for remaining NaNs
    nanindex = find(isnan(V(1:nc:end)));
    notnanindex = setdiff(1:nNodes, nanindex);
    if ~isempty(nanindex) && ~isempty(notnanindex)
        for c = 1:nc
            Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1), coordinatesFEM(notnanindex,2), ...
                V(nc*notnanindex - (nc-c)), 'natural', 'nearest');
            vals = Ftemp(coordinatesFEM(:,1), coordinatesFEM(:,2));
            V(c:nc:end) = vals(:);
        end
    elseif ~isempty(nanindex) && isempty(notnanindex)
        warning('fill_nan_rbf:allNaN', 'All nodes are NaN, cannot interpolate.');
    end
end
