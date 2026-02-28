function [FStraintemp, FStrainWorld] = compute_strain(ULocal, FLocal, coordinatesFEM, DICmesh, DICpara, Df, Dg, Rad)
% Compute strain on a quadtree mesh
%
% Inputs:
%   ULocal         - displacement field [2*nNode x 1]
%   FLocal         - deformation gradient field [4*nNode x 1] (used by cases 0/3)
%   coordinatesFEM - FEM node coordinates [nNode x 2]
%   DICmesh        - mesh structure (.elementMinSize, .markCoordHoleEdge)
%   DICpara        - parameters (.MethodToComputeStrain, .winstepsize, .StrainType,
%                    .ImgRefMask, .smoothness, .DoYouWantToSmoothOnceMore)
%   Df             - reference image structure (.ImgRefMask, .imgSize)
%   Dg             - deformed image structure (.imgSize)
%   Rad            - half window size for plane fitting (case 2); can be [] for other cases
%
% Outputs:
%   FStraintemp    - strain field after StrainType conversion [4*nNode x 1]
%   FStrainWorld   - world-coordinate strain (F21/F12 signs flipped) [4*nNode x 1]
%
% ----------------------------------
% switch MethodToComputeStrain
%   case 3: finite element Gauss points;
% ----------------------------------------------
% Author: Jin Yang.
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch DICpara.MethodToComputeStrain

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 0 % ALDIC directly solved deformation gradients
        FSubpb2 = FLocal;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 1 % Central finite difference

        % Compute strain method I: Use Finite difference operator or FEM solver
        minCoordStep = min( [DICmesh.elementMinSize] );
        xList = [ceil(min(coordinatesFEM(:,1))) : minCoordStep : floor(max(coordinatesFEM(:,1)))]';
        yList = [ceil(min(coordinatesFEM(:,2))) : minCoordStep : floor(max(coordinatesFEM(:,2)))]';

        [xGrid,yGrid] = ndgrid(xList, yList);
        uGrid = gridfit( coordinatesFEM(:,1), coordinatesFEM(:,2), ULocal(1:2:end), xList, yList,'regularizer','springs' ); uGrid=uGrid';
        vGrid = gridfit( coordinatesFEM(:,1), coordinatesFEM(:,2), ULocal(2:2:end), xList, yList ,'regularizer','springs'); vGrid=vGrid';

        for tempi = 1:length(uGrid(:))
           tempx = xGrid(tempi); tempy = yGrid(tempi);
           if  Df.ImgRefMask(tempx,tempy) == 0
               uGrid(tempi) = nan;
               vGrid(tempi) = nan;
           end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ====== Central finite difference ======
        % uvGrid = [uGrid(:),vGrid(:)]'; uvGrid=uvGrid(:);
        %
        % D = funDerivativeOp(length(xList),length(yList),minCoordStep);
        % FStrainGrid = D*uvGrid(:);
        %
        % figure, surf(reshape(FStrainGrid(1:4:end),length(xList),length(yList)),'edgecolor','none');
        % view(2);  axis equal; axis tight; colorbar; caxis([-0.2,0.2])

        % Try to use regularization
        % tempF11 = regularizeNd([xGrid(:),yGrid(:)],FStrainGrid(1:4:end),{xList,yList},DICpara.smoothness);
        % tempF21 = regularizeNd([xGrid(:),yGrid(:)],FStrainGrid(2:4:end),{xList,yList},DICpara.smoothness);
        % tempF12 = regularizeNd([xGrid(:),yGrid(:)],FStrainGrid(3:4:end),{xList,yList},DICpara.smoothness);
        % tempF22 = regularizeNd([xGrid(:),yGrid(:)],FStrainGrid(4:4:end),{xList,yList},DICpara.smoothness);
        % figure, surf(tempF22,'edgecolor','none');  view(2);  axis equal; axis tight; colorbar; caxis([-0.2,0.2])


        Rad = 1*ceil(mean(DICpara.winstepsize)/minCoordStep/2);
        if isfield(DICpara, 'showPlots'), showWB = DICpara.showPlots; else, showWB = true; end
        [UNew,dudx,dudy,iGrid,jGrid] = plane_fit(reshape(uGrid,length(xList),length(yList)), minCoordStep, minCoordStep, Rad, showWB);
        [VNew,dvdx,dvdy,iGrid,jGrid] = plane_fit(reshape(vGrid,length(xList),length(yList)), minCoordStep, minCoordStep, Rad, showWB);

        [row,col] = find(isnan(VNew)==1);
        nonNanIndtemp = sub2ind([length(xList)-2*Rad,length(yList)-2*Rad], row,col);
        nonNanInd = sub2ind([length(xList),length(yList)], iGrid(nonNanIndtemp),jGrid(nonNanIndtemp));
        F_F11 = scatteredInterpolant(xGrid(nonNanInd),yGrid(nonNanInd),dudx(nonNanIndtemp));
        F11 = F_F11(coordinatesFEM(:,1),coordinatesFEM(:,2));
        F_F21 = scatteredInterpolant(xGrid(nonNanInd),yGrid(nonNanInd),dvdx(nonNanIndtemp));
        F21 = F_F21(coordinatesFEM(:,1),coordinatesFEM(:,2));
        F_F12 = scatteredInterpolant(xGrid(nonNanInd),yGrid(nonNanInd),dudy(nonNanIndtemp));
        F12 = F_F12(coordinatesFEM(:,1),coordinatesFEM(:,2));
        F_F22 = scatteredInterpolant(xGrid(nonNanInd),yGrid(nonNanInd),dvdy(nonNanIndtemp));
        F22 = F_F22(coordinatesFEM(:,1),coordinatesFEM(:,2));

        FSubpb2 = [F11(:),F21(:),F12(:),F22(:)]'; FSubpb2 = FSubpb2(:);
        %for tempk=0:3
        %    FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FSubpb1(4*DICmesh.markCoordHoleEdge-tempk);
        %end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 2 % Plane fitting

        % {coordinatesFEM, ULocal} --> {F}
        USubpb2 = nan*ULocal;
        FSubpb2 = repmat(USubpb2,2,1);

        % Compute strain method II: Use Plane Fitting method
        % prompt = 'What is your half window size (unit: px): ';
        % Rad = input(prompt);

        % dilatedI = ( imgaussfilt(double(Dg.ImgRefMask), 0.5) );
        % dilatedI = logical( dilatedI > 0.1);
        dilatedI = logical(DICpara.ImgRefMask); % figure, imshow(dilatedI);
        cc = bwconncomp(dilatedI,8);
        [row1,col1] = find(round(coordinatesFEM(:,1))>1);
        [row2,col2] = find(round(coordinatesFEM(:,1))<Dg.imgSize(1));
        [row3,col3] = find(round(coordinatesFEM(:,2))>1);
        [row4,col4] = find(round(coordinatesFEM(:,2))<Dg.imgSize(2));
        row1234 = intersect(intersect(intersect(row1,row2),row3),row4);

        indPxAll = sub2ind( Dg.imgSize, round(coordinatesFEM(row1234,1)), round(coordinatesFEM(row1234,2)) );
        % figure, plot(round(coordinatesFEM(row1234,1)),round(coordinatesFEM(row1234,2)),'.');
        % figure, plot(stats(tempi).PixelList(:,1), stats(tempi).PixelList(:,2),'.');
        stats = regionprops(cc,'Area','PixelList');
        for tempi = 1:length(stats)

            try

                %%%%% Find those nodes %%%%%
                indPxtempi = sub2ind( Dg.imgSize, stats(tempi).PixelList(:,2), stats(tempi).PixelList(:,1) );
                Lia = ismember(indPxAll,indPxtempi); [LiaList,~] = find(Lia==1);
                % figure, plot(coordinatesFEM(LiaList,1),coordinatesFEM(LiaList,2),'.');
                uv = [ULocal(2*row1234(LiaList)-1), ULocal(2*row1234(LiaList))];
                [~,UPlaneFittemp,FPlaneFittemp] = comp_def_grad(uv, coordinatesFEM(row1234(LiaList),1:2), Rad, 1e6, []);

                % figure, plot3( coordinatesFEM(row1234(LiaList),1), coordinatesFEM(row1234(LiaList),2), ULocal(2*row1234(LiaList)-1), '.' );
                % figure, plot3(coordinatesFEM(:,1), coordinatesFEM(:,2), USubpb2(1:2:end),'.');

                for tempj = 1:4
                    FSubpb2(4*row1234(LiaList)-4+tempj) = FPlaneFittemp(tempj:4:end);
                end
                for tempj = 1:2
                    USubpb2(2*row1234(LiaList)-2+tempj) = UPlaneFittemp(tempj:2:end);
                end
                % figure, plot3(coordinatesFEM(:,1), coordinatesFEM(:,2), FSubpb2(1:4:end),'.');

            catch
            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Fill nans %%%%%%
%         nanindex = find(isnan(USubpb2(1:2:end))==1); notnanindex = setdiff(row1234,nanindex);
%         nanindexF = find(isnan(FSubpb2(1:4:end))==1); notnanindexF = setdiff(row1234,nanindexF);
%
%         if ~isempty(nanindex) || ~isempty(nanindexF)
%
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),USubpb2(2*notnanindex-1),'nearest','nearest');
%             U1 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindex,1),coordinatesFEM(notnanindex,2),USubpb2(2*notnanindex),'nearest','nearest');
%             U2 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),FSubpb2(4*notnanindexF-3),'nearest','nearest');
%             F11 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),FSubpb2(4*notnanindexF-2),'nearest','nearest');
%             F21 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),FSubpb2(4*notnanindexF-1),'nearest','nearest');
%             F12 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%             Ftemp = scatteredInterpolant(coordinatesFEM(notnanindexF,1),coordinatesFEM(notnanindexF,2),FSubpb2(4*notnanindexF-0),'nearest','nearest');
%             F22 = Ftemp(coordinatesFEM(row1234,1),coordinatesFEM(row1234,2));
%
%             USubpb2(2*row1234-1) = U1; USubpb2(2*row1234) = U2;
%             FSubpb2(4*row1234-3) = F11; FSubpb2(4*row1234-2) = F21;
%             FSubpb2(4*row1234-1) = F12; FSubpb2(4*row1234) = F22;
%
%         end

        % ULocal=USubpb2;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 3 % Direct output or finite element method

        FSubpb2 = FLocal;
        try
            if DICpara.DoYouWantToSmoothOnceMore == 0
                FSubpb2 = smooth_strain_rbf(FSubpb2,DICmesh,DICpara);
                for tempk=0:3
                    FSubpb2(4*DICmesh.markCoordHoleEdge-tempk) = FLocal(4*DICmesh.markCoordHoleEdge-tempk);
                end
            end
        catch
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    otherwise

        disp('Wrong Input to compute strain field!')

end




%% Update infinitesimal strain to other finite strains
FStrain = FSubpb2;
FStrainFinite = FStrain;
for tempi = 1:4:length(FStrain)

    % Obtain each component of def grad tensor
    dudx = FStrain(tempi);
    dvdx = FStrain(tempi+1);
    dudy = FStrain(tempi+2);
    dvdy = FStrain(tempi+3);

    switch DICpara.StrainType
        case 0 % Infinitesimal stran
            % Do nothing
        case 1 % Eluerian strain
            FStrainFinite(tempi) = 1/(1-dudx)-1;
            FStrainFinite(tempi+3) = 1/(1-dvdy)-1;
            FStrainFinite(tempi+2) = dudy/(1-dvdy);
            FStrainFinite(tempi+1) = dvdx/(1-dudx);
        case 2 % Green-Lagrangian strain: E=(C-I)/2
            FStrainFinite(tempi) = 0.5*(dudx*2-dudx^2-dvdx^2);
            FStrainFinite(tempi+3) = 0.5*(dvdy*2-dudy^2-dvdy^2);
            FStrainFinite(tempi+2) = 0.5*(dudy+dvdx-dudx*dudy-dvdx*dvdy);
            FStrainFinite(tempi+1) = 0.5*(dvdx+dudy-dudy*dudx-dvdy*dvdx);
        case 3
            disp('Press "Ctrl+C" to modify by yourself.'); pause;
        otherwise
            disp('Wrong strain type!');
    end

end

FStraintemp = FStrainFinite;
FStrainWorld = FStraintemp; FStrainWorld(2:4:end) = -FStrainWorld(2:4:end); FStrainWorld(3:4:end) = -FStrainWorld(3:4:end);
