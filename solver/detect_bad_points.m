function [badPts, badPtNum] = detect_bad_points(ConvItPerEle, maxIterNum, coordinatesFEM, sigmaFactor, minThreshold)
%DETECT_BAD_POINTS  Identify ICGN subsets that failed or converged abnormally.
%
%   [badPts, badPtNum] = detect_bad_points(ConvItPerEle, maxIterNum, coordinatesFEM, sigmaFactor, minThreshold)
%
%   Finds subsets that:
%     - have negative convergence count (try/catch failure)
%     - exceeded maxIterNum-1 iterations
%     - converged significantly slower than the population mean
%
%   INPUTS:
%     ConvItPerEle   - nNodes x 1 convergence iteration counts
%     maxIterNum     - maximum iteration limit
%     coordinatesFEM - nNodes x 2 node coordinates (for counting)
%     sigmaFactor    - outlier sigma multiplier (default 0.25 for subpb1, 1.0 for local_icgn)
%     minThreshold   - minimum iteration threshold (default 10 for subpb1, 6 for local_icgn)
%
%   OUTPUTS:
%     badPts   - indices of bad subsets
%     badPtNum - count excluding mask-only failures (maxIterNum+2)

    nNodes = size(coordinatesFEM, 1);

    [row1,~] = find(ConvItPerEle(:) < 0);
    [row2,~] = find(ConvItPerEle(:) > maxIterNum-1);
    [row3,~] = find(ConvItPerEle(:) == maxIterNum+2);
    badPts = unique(union(row1, row2));
    badPtNum = length(badPts) - length(row3);

    % Statistical outlier detection on "good" points
    goodPts = setdiff(1:nNodes, badPts);
    if ~isempty(goodPts)
        if sigmaFactor == 1.0
            % local_icgn uses fitdist for robust estimation
            pd = fitdist(ConvItPerEle(goodPts), 'Normal');
            mu_val = pd.mu;
            sigma_val = pd.sigma;
        else
            mu_val = mean(ConvItPerEle(goodPts));
            sigma_val = std(ConvItPerEle(goodPts));
        end
        [row4,~] = find(ConvItPerEle(:) > max([mu_val + sigmaFactor*sigma_val, minThreshold]));
        badPts = unique(union(badPts, row4));
    end
end
