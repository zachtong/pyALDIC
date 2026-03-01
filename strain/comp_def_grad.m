function [XY,U,F] = comp_def_grad(uv, parCoord, search_radius_px, n_neighbors, ImgMask)
%comp_def_grad: to compute tracked displacements and deformation gradient
%                 tensor based on the moving least square fitting method
%   [XY,U,F] = comp_def_grad(uv, parCoord, search_radius_px, n_neighbors, ImgMask)
% ---------------------------------------------------
%
%   INPUT:  uv                      Tracked displacement components [n x 2]
%           parCoord                Coordinates of particles
%           search_radius_px        plane-fit search radius [px]
%           n_neighbors             (unused, kept for backward compatibility)
%           ImgMask                 Image mask (optional, can be [])

%   OUTPUT: XY                      Coordinates of particles in image A
%           U                       Disp vectoc U = [U_node1, V_node1, ..., U_nodeN, V_nodeN]';
%           F                       Deformation gradient tensor is assembled into a long vector:
%                                   F = [F11_node1, F21_node1, F12_node1, F22_node1,
%                                        ...,
%                                        F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%
% ---------------------------------------------------
% Author: Jin Yang
% Contact and support: jyang526@wisc.edu
% Date: 2020.12.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
U = nan(size(uv,1)*2,1); F = repmat(U,2,1); XY = nan(size(uv,1),2);

%%
% Calculate neighbourhood indices within radius search_radius_px (rangesearch: O(N log N))
neighborInd = rangesearch(parCoord(:,1:2), parCoord(:,1:2), search_radius_px);

for parInd = 1:size(parCoord,1)

    parNeighborInd = neighborInd{parInd};

    try

        if ~isempty(ImgMask) % Use image mask file

            tempfImgMask = ImgMask( round([parCoord(parInd,1)-search_radius_px : parCoord(parInd,1)+search_radius_px]) , ...
                                round([parCoord(parInd,2)-search_radius_px : parCoord(parInd,2)+search_radius_px]) );

            tempf_BW2 = bwselect(logical(tempfImgMask), search_radius_px+1, search_radius_px+1, 4 );

            % Only use neighbors that are single-connected
            parNeighborInd2 = sub2ind( (2*search_radius_px+1)*[1,1], round(parCoord(parNeighborInd,1)-(parCoord(parInd,1)-search_radius_px)+1), ...
               round(parCoord(parNeighborInd,2)-(parCoord(parInd,2)-search_radius_px)+1)  );

            [parNeighborInd3,~] = find(tempf_BW2(parNeighborInd2)>0);
            parNeighborInd = parNeighborInd(parNeighborInd3);

        end
   
        if length(parNeighborInd) > 3
        % Coordinates: part_A(parNeighborInd, 1:2)
        % u: uvw(parNeighborInd, 1)
        % v: uvw(parNeighborInd, 2)
        %  
        % % We need at least four data points to compute F = {u,x v,x u,y v,y}
        %   x = x0 + u + u,x*(x-x0) + u,y*(y-y0)   
        %   y = y0 + v + v,x*(x-x0) + v,y*(y-y0)  
        
         
        % clf, plot3(parCoord(parNeighborInd,1)-parCoord(parInd,1), parCoord(parNeighborInd,2)-parCoord(parInd,2), ...
        % uv(parNeighborInd,2), '.'); 
         
         
        AMatrix = [ones(length(parNeighborInd),1), parCoord(parNeighborInd,1)-parCoord(parInd,1), ...
                                                   parCoord(parNeighborInd,2)-parCoord(parInd,2) ];
         
        %%%%% Weighted least square fitting (Gaussian weight) %%%%%
        WeightMatrix = diag(exp(-((AMatrix(:,2)/search_radius_px).^2+(AMatrix(:,3)/search_radius_px).^2)));

        UPara = (WeightMatrix * AMatrix) \ (WeightMatrix * uv(parNeighborInd,1));
        VPara = (WeightMatrix * AMatrix) \ (WeightMatrix * uv(parNeighborInd,2));
           
        U(2*parInd-1:2*parInd) = [UPara(1); VPara(1)];
        F(4*parInd-3:4*parInd) = reshape([UPara(2:3)'; VPara(2:3)'],4,1); 
        XY(parInd,1:2) = parCoord(parInd,1:2);
        
        else
            
        U(2*parInd-1:2*parInd) = [uv(parInd,1);uv(parInd,2)];
        F(4*parInd-3:4*parInd) = nan(4,1);
        XY(parInd,1:2) = parCoord(parInd,1:2);
        
        end
        
    catch

        U(2*parInd-1:2*parInd) = [uv(parInd,1);uv(parInd,2)];
        F(4*parInd-3:4*parInd) = nan(4,1);
        XY(parInd,1:2) = parCoord(parInd,1:2);

    end

end

%% Fill NaN values in F (once, after the loop)
nanindexF = find(isnan(F(1:4:end))==1); notnanindexF = setdiff(1:size(F,1)/4, nanindexF);

if ~isempty(nanindexF) && ~isempty(notnanindexF)

    Ftemp = scatteredInterpolant(parCoord(notnanindexF,1),parCoord(notnanindexF,2),F(4*notnanindexF-3),'linear','linear');
    F11 = Ftemp(parCoord(:,1),parCoord(:,2));
    Ftemp = scatteredInterpolant(parCoord(notnanindexF,1),parCoord(notnanindexF,2),F(4*notnanindexF-2),'linear','linear');
    F21 = Ftemp(parCoord(:,1),parCoord(:,2));
    Ftemp = scatteredInterpolant(parCoord(notnanindexF,1),parCoord(notnanindexF,2),F(4*notnanindexF-1),'linear','linear');
    F12 = Ftemp(parCoord(:,1),parCoord(:,2));
    Ftemp = scatteredInterpolant(parCoord(notnanindexF,1),parCoord(notnanindexF,2),F(4*notnanindexF-0),'linear','linear');
    F22 = Ftemp(parCoord(:,1),parCoord(:,2));

    F = [F11(:),F21(:),F12(:),F22(:)]'; F = F(:);

end



