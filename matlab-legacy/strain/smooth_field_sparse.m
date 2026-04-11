function V = smooth_field_sparse(V, coordinatesFEM, DICpara, nComponents, smoothness, nodeRegionMap)
%SMOOTH_FIELD_SPARSE  Smooth a nodal field using sparse Gaussian kernel.
%
%   V = smooth_field_sparse(V, coordinatesFEM, DICpara, nComponents, smoothness)
%   V = smooth_field_sparse(V, coordinatesFEM, DICpara, nComponents, smoothness, nodeRegionMap)
%
%   Applies per-connected-region Gaussian-weighted averaging using rangesearch.
%   O(N log N) complexity, replaces O(N^3) RBF smoothing.
%
%   INPUTS:
%     V              - column vector, interleaved components
%     coordinatesFEM - nNodes x 2 coordinates
%     DICpara        - struct with .ImgRefMask, .ImgSize, .winstepsize
%     nComponents    - 2 (displacement) or 4 (strain/deformation gradient)
%     smoothness     - smoothing parameter (from DICpara.DispSmoothness or StrainSmoothness)
%     nodeRegionMap  - (optional) precomputed struct from precompute_node_regions
%
%   OUTPUT:
%     V              - smoothed field (same format as input)

if smoothness < 1e-8
    return;  % no smoothing requested
end

V = full(V);

% Map smoothness parameter to Gaussian kernel width
h = min(DICpara.winstepsize);
sigma = h * max(0.3, 500 * smoothness);
R = 3 * sigma;

% Get node-to-region mapping (precomputed or on-the-fly)
if nargin >= 6 && ~isempty(nodeRegionMap)
    regionNodeLists = nodeRegionMap.regionNodeLists;
    nRegions = nodeRegionMap.nRegions;
else
    dilatedI = logical(DICpara.ImgRefMask);
    cc = bwconncomp(dilatedI, 8);
    indPxAll = sub2ind(DICpara.ImgSize, round(coordinatesFEM(:,1)), round(coordinatesFEM(:,2)));
    stats = regionprops(cc, 'Area', 'PixelList');
    regionNodeLists = {};
    nRegions = 0;
    for tempi = 1:length(stats)
        if stats(tempi).Area > 20
            indPxtempi = sub2ind(DICpara.ImgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1));
            Lia = ismember(indPxAll, indPxtempi);
            LiaList = find(Lia);
            if length(LiaList) >= 2
                nRegions = nRegions + 1;
                regionNodeLists{nRegions} = LiaList; %#ok<AGROW>
            end
        end
    end
end

for r = 1:nRegions
    LiaList = regionNodeLists{r};
    nRegion = length(LiaList);
    coords_region = coordinatesFEM(LiaList, :);

    % Find neighbors within radius R
    [idx, dist] = rangesearch(coords_region, coords_region, R);

    % Build sparse Gaussian weight matrix
    nnzTotal = sum(cellfun(@length, idx));
    rows = zeros(nnzTotal, 1);
    cols = zeros(nnzTotal, 1);
    vals = zeros(nnzTotal, 1);
    pos = 0;
    for i = 1:nRegion
        neighbors = idx{i};
        n = length(neighbors);
        w = exp(-dist{i}.^2 / (2 * sigma^2));
        w = w / sum(w);
        rows(pos+1:pos+n) = i;
        cols(pos+1:pos+n) = neighbors(:);
        vals(pos+1:pos+n) = w(:);
        pos = pos + n;
    end
    W = sparse(rows, cols, vals, nRegion, nRegion);

    % Apply smoothing: one sparse matrix-vector multiply per component
    for c = 1:nComponents
        globalIdx = nComponents * LiaList - (nComponents - c);
        localVals = V(globalIdx);
        nanMask = isnan(localVals);
        if all(nanMask), continue; end
        if any(nanMask)
            localVals(nanMask) = 0;
            nanCols = find(nanMask);
            W_adj = W;
            W_adj(:, nanCols) = 0;
            rowSums = full(sum(W_adj, 2));
            rowSums(rowSums < eps) = 1;
            W_adj = spdiags(1./rowSums, 0, nRegion, nRegion) * W_adj;
            smoothed = W_adj * localVals;
            smoothed(nanMask) = NaN;
        else
            smoothed = W * localVals;
        end
        V(globalIdx) = smoothed;
    end
end

end
