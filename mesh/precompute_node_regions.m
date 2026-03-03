function nodeRegionMap = precompute_node_regions(coordinatesFEM, DICpara)
%PRECOMPUTE_NODE_REGIONS  Map each FEM node to its mask connected region.
%
%   nodeRegionMap = precompute_node_regions(coordinatesFEM, DICpara)
%
%   Performs bwconncomp + regionprops once on the reference mask, then
%   maps each FEM node to its enclosing connected region via ismember.
%   The result can be passed to smooth_field_sparse / smooth_disp_rbf /
%   smooth_strain_rbf to avoid repeating this O(N) mask analysis on
%   every smoothing call.
%
%   INPUTS:
%     coordinatesFEM - nNodes x 2, FEM node coordinates (row, col)
%     DICpara        - struct with .ImgRefMask and .ImgSize
%
%   OUTPUT:
%     nodeRegionMap  - struct with fields:
%       .regionNodeLists  cell {nRegions x 1}, each = node indices in that region
%       .nRegions         number of connected regions with >= 2 nodes

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

nodeRegionMap.regionNodeLists = regionNodeLists;
nodeRegionMap.nRegions = nRegions;

end
