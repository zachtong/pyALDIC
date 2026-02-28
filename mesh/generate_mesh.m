function [DICmesh, U0] = generate_mesh(DICmesh, DICpara, Df, U0)
% Generate a quadtree mesh considering sample's complex geometry
%
% Inputs:
%   DICmesh - mesh structure (.coordinatesFEM, .elementsFEM, .elementMinSize)
%   DICpara - parameters (.ImgRefMask, .winstepsize, .winsize)
%   Df      - reference image structure (.ImgRefMask, .imgSize)
%   U0      - initial displacement field on uniform mesh [2*nNode x 1]
%
% Outputs:
%   DICmesh - updated mesh structure (new .coordinatesFEM, .elementsFEM,
%             .irregular, .markCoordHoleEdge, .dirichlet, .coordinatesFEMWorld)
%   U0      - displacement field interpolated onto quadtree mesh
%
% ----------------------------------------------
% References
% [1] J Yang, K Bhattacharya. Fast adaptive mesh augmented Lagrangian Digital Image
% Correlation. Under review.
% [2] S Funken, A Schmidt. Adaptive mesh refinement in 2D: an efficient
% implementation in MATLAB. Comp. Meth. Appl. Math. 20:459-479, 2020.
% [3] Rbfinterp. Matlab File Exchange open source.
% https://www.mathworks.com/matlabcentral/fileexchange/10056-scattered-data-interpolation-and-approximation-using-radial-base-functions
% ----------------------------------------------
% Author: Jin Yang
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================


disp('--- Generate a quadtree mesh ---')
%% ====== Remove finite elements where there is a hole ======
coordinatesFEMQuadtree = DICmesh.coordinatesFEM;
elementsFEMQuadtree = DICmesh.elementsFEM(:,1:4);
irregular = zeros(0,3);

while 1 % Generate a Quadtree mesh
    [~,mark4] = mark_edge(coordinatesFEMQuadtree,elementsFEMQuadtree,DICpara.ImgRefMask,DICmesh.elementMinSize); % Don't delete "*2"
    mark4 = find(mark4);
    [coordinatesFEMQuadtree,elementsFEMQuadtree,irregular] = qrefine_r(coordinatesFEMQuadtree,elementsFEMQuadtree,irregular,mark4);
    if isempty(mark4)
        break
    end
end

%%%%% Re-order node index in elements %%%%%
for tempj = 1:size(elementsFEMQuadtree,1)
    coordxAll = coordinatesFEMQuadtree(elementsFEMQuadtree(tempj,1:4),1);
    coordyAll = coordinatesFEMQuadtree(elementsFEMQuadtree(tempj,1:4),2);
    coordxAll_sorted = sort(coordxAll(:));
    coordyAll_sorted = sort(coordyAll(:));

    temp1 = (coordxAll-coordxAll_sorted(1)).^2 + 2*((coordyAll-coordyAll_sorted(1))).^2 ;
    [temp2,temp3] = sort(temp1);

    elementsFEMQuadtree(tempj,1:4) = elementsFEMQuadtree(tempj,temp3([1,2,4,3]));

    % coordinatesFEMQuadtree(elementsFEMQuadtree(tempj,1:4),1:2)
end


%% %%%%% Plot refined mesh %%%%%
% figure; patch('Faces', elementsFEMQuadtree(:,1:4), 'Vertices', coordinatesFEMQuadtree, 'Facecolor','none','linewidth',1)
% axis equal; axis tight; set(gca,'fontsize',18); set(gcf,'color','w'); box on;
% xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
% title('Quadtree mesh','Interpreter','latex');
% a = gca; a.TickLabelInterpreter = 'latex';

% Update the quadtree mesh to deal with hanging nodes
for tempj = 1:size(irregular,1)
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,1:2), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,8)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,2:3), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,5)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,3:4), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,6)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,[4,1]), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,7)=irregular(tempj,3); end

    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,[2,1]), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,8)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,[3,2]), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,5)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,[4,3]), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,6)=irregular(tempj,3); end
    [Lia, Locb] = ismember( [irregular(tempj,1:2)], elementsFEMQuadtree(:,[1,4]), 'rows' );
    if Lia>0, elementsFEMQuadtree(Locb,7)=irregular(tempj,3); end
end

% Remove elements within the center hole
[markInside4,markOutside4] = mark_inside(coordinatesFEMQuadtree,elementsFEMQuadtree,DICpara.ImgRefMask);
elementsFEMQuadtree = elementsFEMQuadtree(markOutside4,:);


%% Find nodes near the edges

% %%%%% Old codes: Erode ref image mask, and to find elements near holes' edges,
% nhood = [1 1 1; 1 1 1; 1 1 1];
% ImgRefMaskErode = DICpara.ImgRefMask;
% for tempi = 1: floor(0.5*max([20,mean(DICpara.winsize),mean(DICpara.winstepsize)]))-1
%     ImgRefMaskErode = imerode(ImgRefMaskErode, nhood);
% end
% % figure, imshow(ImgRefMaskErode');
% [markEleHoleEdge4,markEleFarOutside4] = mark_inside(coordinatesFEMQuadtree,elementsFEMQuadtree,ImgRefMaskErode);

% %%%%% New codes: Find elements which are refined %%%%%%
elementsFEMQuadtreeSize = sqrt( ( coordinatesFEMQuadtree(elementsFEMQuadtree(:,1),1) - coordinatesFEMQuadtree(elementsFEMQuadtree(:,3),1) ).^2 + ...
        ( coordinatesFEMQuadtree(elementsFEMQuadtree(:,1),2) - coordinatesFEMQuadtree(elementsFEMQuadtree(:,3),2) ).^2 );
[markEleRefine4,~] = find(elementsFEMQuadtreeSize <  0.99*sqrt(2)*max([DICpara.winstepsize,0*DICpara.winsize]));

% %%%%% New codes: Find elements near the boudary %%%%%%
xMin = min(DICmesh.coordinatesFEM(:,1)); xMax = max(DICmesh.coordinatesFEM(:,1));
yMin = min(DICmesh.coordinatesFEM(:,2)); yMax = max(DICmesh.coordinatesFEM(:,2));
[row1,col1] = find(coordinatesFEMQuadtree(elementsFEMQuadtree(:,1),1) < xMin+1.01*DICpara.winstepsize);
[row2,col2] = find(coordinatesFEMQuadtree(elementsFEMQuadtree(:,3),1) > xMax-1.01*DICpara.winstepsize);
[row3,col3] = find(coordinatesFEMQuadtree(elementsFEMQuadtree(:,1),2) < yMin+1.01*DICpara.winstepsize);
[row4,col4] = find(coordinatesFEMQuadtree(elementsFEMQuadtree(:,3),2) > yMax-1.01*DICpara.winstepsize);

markEleHoleEdge4 =  union(row4,union(row3,union(row2,union(row1,markEleRefine4))));
markCoordHoleEdge = unique(elementsFEMQuadtree(markEleHoleEdge4,:));
try
    if markCoordHoleEdge(1)==0, markCoordHoleEdge = markCoordHoleEdge(2:end); end
catch
end

%%%%%% New codes: Find elements near marked elements %%%%%%
for tempi = 1:2 % 2+(round( 32 / mean(DICpara.winstepsize) )^2)

    markEleHoleEdgeNeigh4 = zeros(size(elementsFEMQuadtree,1),1);
    for eleInd = 1:size(elementsFEMQuadtree,1)
        markEleHoleEdgeNeigh4(eleInd) = length(intersect(elementsFEMQuadtree(eleInd,:),markCoordHoleEdge));
    end
    [markEleHoleEdgeNeigh4,~] = find(markEleHoleEdgeNeigh4>0);
    %%%%%%%%%
    markCoordHoleEdge = unique(elementsFEMQuadtree(markEleHoleEdgeNeigh4,:)) ;
    try
        if markCoordHoleEdge(1) == 0, markCoordHoleEdge = markCoordHoleEdge(2:end); end
    catch
    end

end


% %%%%% Store data structure %%%%%
DICmesh.markCoordHoleEdge = markCoordHoleEdge;
DICmesh.dirichlet = DICmesh.markCoordHoleEdge;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%% Plot %%%%%
if isfield(DICpara, 'showPlots') && ~DICpara.showPlots
    % Skip mesh visualization
else
    figure;
    patch('Faces', elementsFEMQuadtree(:,1:4), 'Vertices', coordinatesFEMQuadtree, 'Facecolor','white','linewidth',1);
    patch('Faces', elementsFEMQuadtree(markEleHoleEdge4,1:4), 'Vertices', coordinatesFEMQuadtree, 'Facecolor','yellow','linewidth',1);
    hold on; patch('Faces', elementsFEMQuadtree(markEleHoleEdgeNeigh4,1:4), 'Vertices', coordinatesFEMQuadtree, 'Facecolor','yellow','linewidth',1);
    axis equal; axis tight; set(gca,'fontsize',18); set(gcf,'color','w'); box on;
    xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
    title('Quadtree mesh','Interpreter','latex');
    a = gca; a.TickLabelInterpreter = 'latex';

    lgd = legend('Quadtree mesh elements','Elements near the edge','interpreter','latex','location','northeastoutside');
    set(lgd,'fontsize',13);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialize variable U for the generated quadtree mesh
U0NotNanInd = find(~isnan(U0(1:2:end)));
U0Quadtree = 0*coordinatesFEMQuadtree(:);

dilatedI = ( imgaussfilt(double(Df.ImgRefMask),1) );
dilatedI = logical( dilatedI > 0.01);
cc = bwconncomp(dilatedI, 4);
indPxAll = sub2ind( Df.imgSize, coordinatesFEMQuadtree(:,1), coordinatesFEMQuadtree(:,2) );
indPxNotNanAll = sub2ind( Df.imgSize, DICmesh.coordinatesFEM(U0NotNanInd,1), DICmesh.coordinatesFEM(U0NotNanInd,2) );
stats = regionprops(cc,'Area','PixelList');
for tempi = 1:length(stats)

    try

        %%%%% Find those nodes %%%%%
        indPxtempi = sub2ind( Df.imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
        Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
        Lib = ismember(indPxNotNanAll,indPxtempi); [LibList,~] = find(Lib==1);

        %%%%% RBF (Radial basis function) works better than "scatteredInterpolant" %%%%%
        % ------ Disp u ------
        op1 = rbfcreate( [DICmesh.coordinatesFEM(U0NotNanInd(LibList),1:2)]',[U0(2*U0NotNanInd(LibList)-1)]','RBFFunction', 'thinplate'); rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEMQuadtree(LiaList,1:2)]', op1);
        U0Quadtree(2*LiaList-1) = fi1(:);

        % ------ Disp v ------
        op1 = rbfcreate( [DICmesh.coordinatesFEM(U0NotNanInd(LibList),1:2)]',[U0(2*U0NotNanInd(LibList))]','RBFFunction', 'thinplate'); rbfcheck(op1);
        fi1 = rbfinterp( [coordinatesFEMQuadtree(LiaList,1:2)]', op1);
        U0Quadtree(2*LiaList) = fi1(:);

    catch
    end
end

U0 = U0Quadtree;

% F_dispu = scatteredInterpolant( DICmesh.coordinatesFEM(U0NotNanInd,1),DICmesh.coordinatesFEM(U0NotNanInd,2),U0(2*U0NotNanInd-1),'linear','linear' );
% F_dispv = scatteredInterpolant( DICmesh.coordinatesFEM(U0NotNanInd,1),DICmesh.coordinatesFEM(U0NotNanInd,2),U0(2*U0NotNanInd),'linear','linear' );
%
% U0 = 0*coordinatesFEMQuadtree(:);
% temp = F_dispu(coordinatesFEMQuadtree(:,1),coordinatesFEMQuadtree(:,2)); U0(1:2:end)=temp(:);
% temp = F_dispv(coordinatesFEMQuadtree(:,1),coordinatesFEMQuadtree(:,2)); U0(2:2:end)=temp(:);

plot_disp_show( -U0,coordinatesFEMQuadtree,elementsFEMQuadtree(:,1:4),DICpara,'NoEdgeColor');

if size(elementsFEMQuadtree,2)<8 % make sure the column# of elementsFEMQuadtree is 8
    elementsFEMQuadtree(:,8) = 0*elementsFEMQuadtree(:,1);
end

DICmesh.coordinatesFEM = coordinatesFEMQuadtree;
DICmesh.elementsFEM = elementsFEMQuadtree;
DICmesh.irregular = irregular;
DICmesh.coordinatesFEMWorld = [DICmesh.coordinatesFEM(:,1),size(DICpara.ImgRefMask,2)+1-DICmesh.coordinatesFEM(:,2)];

% ===== Remove bad points =====
% JY!!!
% [U0,~] = funRemoveOutliersQuadtree(DICmesh,DICpara,U0,[U0;U0]);
