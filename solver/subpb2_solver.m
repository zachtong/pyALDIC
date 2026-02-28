function [Uhat] = subpb2_solver(DICmesh,GaussPtOrder,beta,mu,U,F,udual,vdual, ...
                                 alpha,winstepsize,waitBarDisplayOrNot)
%FUNCTION [Uhat] = subpb2_solver(DICmesh,GaussPtOrder,beta,mu, ...
%          U,F,udual,vdual,alpha,winstepsize,waitBarDisplayOrNot)
% AL-DIC Subproblem 2 is solved over a quadtree mesh to find a globally
% kinematically compatible deformation field by finite element method.
% ----------------------------------------------
% 
%   INPUT: DICmesh             DIC FE Q4 mesh: coordinatesFEM, elementsFEM
%          GaussPtOrder        Gauss point order used in FE Q4 element
%          beta, mu            Two constant coefficients
%          U                   Disp vector: U = [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%          F                   Deformation gradient: F = [F11_node1, F21_node1, F12_node1, F22_node1, ... 
%                                                         ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%          udual,vdual         Dual variables
%          alpha               Smoothness coefficient. Not needed here, i.e., alpha=0
%          winstepsize         DIC FE Q4 mesh spacing
%          waitBarDisplayOrNot Display a waitbar or not 
%
%   OUTPUT: Uhat               Solved globally kinematically compatible displacement field
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 2018.03, 2020.12 
% ==============================================                                  


%% Initialization
coordinatesFEM = DICmesh.coordinatesFEM;
elementsFEM = DICmesh.elementsFEM;
dirichlet = DICmesh.dirichlet;
neumann = DICmesh.neumann;

DIM = 2; 
NodesNumPerEle = 4; 
ClusterNo = 1;
FEMSize = DIM*size(coordinatesFEM,1);


%% ====== Initialize variables ======
fprintf('[debug Subpb2] Input sizes: U=%s, F=%s, udual=%s, vdual=%s, FEMSize=%d\n', ...
    mat2str(size(U)), mat2str(size(F)), mat2str(size(udual)), mat2str(size(vdual)), FEMSize);
Uhat = U; U = [U;zeros(DIM*NodesNumPerEle,1)]; v = [0*vdual;zeros(DIM*NodesNumPerEle,1)];
F = [F;zeros(DIM^2*NodesNumPerEle,1)]; W = [0*udual;zeros(DIM^2*NodesNumPerEle,1)];
fprintf('[debug Subpb2] After padding: U=%s, v=%s, F=%s, W=%s\n', ...
    mat2str(size(U)), mat2str(size(v)), mat2str(size(F)), mat2str(size(W)));
UMinusv = U-v; FMinusW = F-W;

% ====== Gaussian quadrature parameter ======
switch GaussPtOrder
    case 2 % 2*2 Gauss points 
        gqpt1 = -1/sqrt(3); gqpt2 = 1/sqrt(3); gqpt = [gqpt1,gqpt2]; 
        gqwt1 = 1; gqwt2 = 1; gqwt = [gqwt1,gqwt2];
    case 3 % 3*3 Gauss points 
        gqpt1 = 0; gqpt2 = sqrt(3/5); gqpt3 = -sqrt(3/5); gqpt = [gqpt1,gqpt2,gqpt3];
        gqwt1 = 8/9; gqwt2 = 5/9; gqwt3 = 5/9; gqwt = [gqwt1,gqwt2,gqwt3];
    case 4 % 4*4 Gauss points 
        gqpt1 = 0.339981; gqpt2 = -0.339981; gqpt3 = 0.861136; gqpt4 = -0.861136;  
        gqwt1 = 0.652145; gqwt2 = 0.652145; gqwt3 = 0.347855; gqwt4 = 0.347855;
        gqpt = [gqpt1,gqpt2,gqpt3,gqpt4]; gqwt = [gqwt1,gqwt2,gqwt3,gqwt4];
    case 5 % 5*5 Gauss points 
        gqpt1 = 0; gqpt2 = 0.538469; gqpt3 = -0.538469; gqpt4 = 0.90618; gqpt5 = -0.90618;
        gqwt1 = 0.568889; gqwt2 = 0.478629; gqwt3 = 0.478629; gqwt4 = 0.236927; gqwt5 = 0.236927;
        gqpt = [gqpt1,gqpt2,gqpt3,gqpt4,gqpt5]; gqwt = [gqwt1,gqwt2,gqwt3,gqwt4,gqwt5];
    otherwise
        disp('Not implemented yet!')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== Initialize Global FEM solver ======
ConvergeOrNot = 0; IterStep = 0;
while ConvergeOrNot < 0.5 && IterStep < 1 % Only do one step; IterStep will add value "1" in the first iteration.
    
    IterStep = IterStep+1;
    
    if waitBarDisplayOrNot == 0
        if (ClusterNo==0) || (ClusterNo==1)
            hbar = waitbar(0, ['FE-method to solve Subproblem 2.']);
        else
            hbar=parfor_progressbar(size(elementsFEM,1),['FE-method to solve Subproblem 2.']);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ============= For each element, assemble the stiffness matrix ============
    for eleInd = 1:size(elementsFEM,1) % eleInd is the element index
         
        if waitBarDisplayOrNot == 0
            if ClusterNo== 0 || ClusterNo== 1
                waitbar(eleInd/size(elementsFEM,1));
            else
                hbar.iterate(1);
            end
        end
        
        tempA = zeros(DIM*8,DIM*8); tempb = tempA(:,1);
        
        % ------ Find four corner points ------
        pt1x = coordinatesFEM(elementsFEM(eleInd,1),1); pt1y = coordinatesFEM(elementsFEM(eleInd,1),2);
        pt2x = coordinatesFEM(elementsFEM(eleInd,2),1); pt2y = coordinatesFEM(elementsFEM(eleInd,2),2);
        pt3x = coordinatesFEM(elementsFEM(eleInd,3),1); pt3y = coordinatesFEM(elementsFEM(eleInd,3),2);
        pt4x = coordinatesFEM(elementsFEM(eleInd,4),1); pt4y = coordinatesFEM(elementsFEM(eleInd,4),2);
        
        % ------ Find mid points 5/6/7/8 -------
        if elementsFEM(eleInd,5) ~= 0
            pt5x = coordinatesFEM(elementsFEM(eleInd,5),1); pt5y = coordinatesFEM(elementsFEM(eleInd,5),2);
        else, pt5x = 0; pt5y = 0; 
        end
        if elementsFEM(eleInd,6) ~= 0
            pt6x = coordinatesFEM(elementsFEM(eleInd,6),1); pt6y = coordinatesFEM(elementsFEM(eleInd,6),2);
        else, pt6x = 0; pt6y = 0; 
        end
        if elementsFEM(eleInd,7) ~= 0
            pt7x = coordinatesFEM(elementsFEM(eleInd,7),1); pt7y = coordinatesFEM(elementsFEM(eleInd,7),2);
        else, pt7x = 0; pt7y = 0; 
        end
        if elementsFEM(eleInd,8) ~= 0
            pt8x = coordinatesFEM(elementsFEM(eleInd,8),1); pt8y = coordinatesFEM(elementsFEM(eleInd,8),2);
        else, pt8x = 0; pt8y = 0; 
        end
        
        % ------ Calculate ksi and eta ------
        % lMatrix = [ point1x*point1y point1x point1y 1;
        %             point2x*point2y point2x point2y 1;
        %             point3x*point3y point3x point3y 1;
        %             point4x*point4y point4x point4y 1 ];
          
        % % ------ Find the element nodal indices ------
        % lb = [-1;1;1;-1]; l = linsolve(lMatrix,lb);
        % mb = [-1;-1;1;1]; m = linsolve(lMatrix,mb);
        
        % ------ Find the element nodal indices ------
        % tempIndexU = [2*elementsFEM(eleInd,1)-1 2*elementsFEM(eleInd,1) 2*elementsFEM(eleInd,2)-1 2*elementsFEM(eleInd,2) ...
        %         2*elementsFEM(eleInd,3)-1 2*elementsFEM(eleInd,3) 2*elementsFEM(eleInd,4)-1 2*elementsFEM(eleInd,4) ...
        %         2*elementsFEM(eleInd,5)-1 2*elementsFEM(eleInd,5) 2*elementsFEM(eleInd,6)-1 2*elementsFEM(eleInd,6) ...
        %         2*elementsFEM(eleInd,7)-1 2*elementsFEM(eleInd,7) 2*elementsFEM(eleInd,8)-1 2*elementsFEM(eleInd,8)];
        % tp = ones(1,DIM);
        % tempIndexU = 2*elementsFEM(eleInd,[tp,2*tp,3*tp,4*tp,5*tp,6*tp,7*tp,8*tp]);
        % tempIndexU(1:2:end) = tempIndexU(1:2:end)-1;  % size of tempIndexU: 1*16
        tempIndexU = 2*elementsFEM(eleInd,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8]);
        tempIndexU(1:2:end) = tempIndexU(1:2:end)-1;
        tempIndexF = [4*elementsFEM(eleInd,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8])'-3, ...
                      4*elementsFEM(eleInd,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8])'-1, ...
                      4*elementsFEM(eleInd,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8])'-2, ...
                      4*elementsFEM(eleInd,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8])'];
                  
        % We don't want temp <= 0, instead, put them to the end
        for tempi = 5:8
            if elementsFEM(eleInd,tempi) == 0
                tempIndexU(2*tempi-1)   = 2*(FEMSize/DIM+1)-1;   tempIndexU(2*tempi)   = 2*(FEMSize/DIM+1);
                tempIndexF(2*tempi-1,1) = 4*(FEMSize/DIM+1)-3;   tempIndexF(2*tempi,1) = 4*(FEMSize/DIM+1)-3;
                tempIndexF(2*tempi-1,2) = 4*(FEMSize/DIM+1)-1;   tempIndexF(2*tempi,2) = 4*(FEMSize/DIM+1)-1;
                tempIndexF(2*tempi-1,3) = 4*(FEMSize/DIM+1)-2;   tempIndexF(2*tempi,3) = 4*(FEMSize/DIM+1)-2;
                tempIndexF(2*tempi-1,4) = 4*(FEMSize/DIM+1);     tempIndexF(2*tempi,4) = 4*(FEMSize/DIM+1);
            end
        end
          
        %% ------ Calculate ksi and eta ------
        % for tempi = 1:size(pointOfx,1)
        %     for tempj = 1:size(pointOfy,1)
        %        ksi = l(1)*pointOfx(tempi)*pointOfy(tempj) + l(2)*pointOfx(tempi) + l(3)*pointOfy(tempj) + l(4) ;
        %        eta = m(1)*pointOfx(tempi)*pointOfy(tempj) + m(2)*pointOfx(tempi) + m(3)*pointOfy(tempj) + m(4) ;
        % %%%%%% Use following codes %%%%%%
        % ------ Nine Gauss integral points ------
        ksietaList = gqpt; ksietaWeightList = gqwt;
        [ksiAll, etaAll] = ndgrid(ksietaList, ksietaList);
        [weightksiAll, weightetaAll] = ndgrid(ksietaWeightList, ksietaWeightList);
        ksiAll = ksiAll(:); etaAll = etaAll(:); weightksiAll = weightksiAll(:); weightetaAll = weightetaAll(:);
         
        % %%%%% Rewrite two for-loops into one for-loop %%%%%
        %for tempi = 1:length(ksietaList)
        %    for tempj = 1:length(ksietaList)
        for tempjj = 1:length(ksiAll)
            
            % ---------------------------
            % ksi = ksietaList(tempi);
            % eta = ksietaList(tempj);
            % weightksi = ksietaWeightList(tempi);
            % weighteta = ksietaWeightList(tempj);
            ksi = ksiAll(tempjj); eta = etaAll(tempjj); weightksi = weightksiAll(tempjj); weighteta = weightetaAll(tempjj);
            
            % ------ Calculate N matrix ------
            deltaPt5 = 0; deltaPt6 = 0; deltaPt7 = 0; deltaPt8 = 0;
            if elementsFEM(eleInd,5) ~= 0; deltaPt5 = 1; end
            if elementsFEM(eleInd,6) ~= 0; deltaPt6 = 1; end
            if elementsFEM(eleInd,7) ~= 0; deltaPt7 = 1; end
            if elementsFEM(eleInd,8) ~= 0; deltaPt8 = 1; end
            
            % N5 = deltaPt5*0.5*(1+ksi)*(1-eta^2); %%%%% Old code %%%%%
            % N6 = deltaPt6*0.5*(1+eta)*(1-ksi^2); %%%%% Old code %%%%%
            % N7 = deltaPt7*0.5*(1-ksi)*(1-eta^2); %%%%% Old code %%%%%
            % N8 = deltaPt8*0.5*(1-eta)*(1-ksi^2); %%%%% Old code %%%%%
            N5 = deltaPt5*0.5*(1+ksi)*(1-abs(eta));
            N6 = deltaPt6*0.5*(1+eta)*(1-abs(ksi));
            N7 = deltaPt7*0.5*(1-ksi)*(1-abs(eta));
            N8 = deltaPt8*0.5*(1-eta)*(1-abs(ksi));
            
            N1 = (1-ksi)*(1-eta)*0.25 - 0.5*(N7+N8);
            N2 = (1+ksi)*(1-eta)*0.25 - 0.5*(N8+N5);
            N3 = (1+ksi)*(1+eta)*0.25 - 0.5*(N5+N6);
            N4 = (1-ksi)*(1+eta)*0.25 - 0.5*(N6+N7);
             
            N = [N1 0 N2 0 N3 0 N4 0 N5 0 N6 0 N7 0 N8 0;
                0 N1 0 N2 0 N3 0 N4 0 N5 0 N6 0 N7 0 N8];
            NDiag = diag([N1,N1,N2,N2,N3,N3,N4,N4,N5,N5,N6,N6,N7,N7,N8,N8]);
            
            % ------ Build Jacobian matrix ------
            % Jacobian matrix J doesn't change with added enriched functions
            J11 = funDN1Dksi(ksi,eta,deltaPt7,deltaPt8)*pt1x + funDN2Dksi(ksi,eta,deltaPt8,deltaPt5)*pt2x + ...
                funDN3Dksi(ksi,eta,deltaPt5,deltaPt6)*pt3x + funDN4Dksi(ksi,eta,deltaPt6,deltaPt7)*pt4x + ...
                funDN5Dksi(ksi,eta,deltaPt5)*pt5x + funDN6Dksi(ksi,eta,deltaPt6)*pt6x + ...
                funDN7Dksi(ksi,eta,deltaPt7)*pt7x + funDN8Dksi(ksi,eta,deltaPt8)*pt8x;
            J12 = funDN1Dksi(ksi,eta,deltaPt7,deltaPt8)*pt1y + funDN2Dksi(ksi,eta,deltaPt8,deltaPt5)*pt2y + ...
                funDN3Dksi(ksi,eta,deltaPt5,deltaPt6)*pt3y + funDN4Dksi(ksi,eta,deltaPt6,deltaPt7)*pt4y + ...
                funDN5Dksi(ksi,eta,deltaPt5)*pt5y + funDN6Dksi(ksi,eta,deltaPt6)*pt6y + ...
                funDN7Dksi(ksi,eta,deltaPt7)*pt7y + funDN8Dksi(ksi,eta,deltaPt8)*pt8y;
            J21 = funDN1Deta(ksi,eta,deltaPt7,deltaPt8)*pt1x + funDN2Deta(ksi,eta,deltaPt8,deltaPt5)*pt2x + ...
                funDN3Deta(ksi,eta,deltaPt5,deltaPt6)*pt3x + funDN4Deta(ksi,eta,deltaPt6,deltaPt7)*pt4x + ...
                funDN5Deta(ksi,eta,deltaPt5)*pt5x + funDN6Deta(ksi,eta,deltaPt6)*pt6x + ...
                funDN7Deta(ksi,eta,deltaPt7)*pt7x + funDN8Deta(ksi,eta,deltaPt8)*pt8x;
            J22 = funDN1Deta(ksi,eta,deltaPt7,deltaPt8)*pt1y + funDN2Deta(ksi,eta,deltaPt8,deltaPt5)*pt2y + ...
                funDN3Deta(ksi,eta,deltaPt5,deltaPt6)*pt3y + funDN4Deta(ksi,eta,deltaPt6,deltaPt7)*pt4y + ...
                funDN5Deta(ksi,eta,deltaPt5)*pt5y + funDN6Deta(ksi,eta,deltaPt6)*pt6y + ...
                funDN7Deta(ksi,eta,deltaPt7)*pt7y + funDN8Deta(ksi,eta,deltaPt8)*pt8y;
            
            J = [J11 J12; J21 J22];
            Jacobian = det(J);
            InvJ = 1/Jacobian*[J22 -J12; -J21 J11];
            
            % ------ Compute DN matrix ------
            DN = [InvJ zeros(2,2); zeros(2,2) InvJ] * ...
                [funDN1Dksi(ksi,eta,deltaPt7,deltaPt8) 0 funDN2Dksi(ksi,eta,deltaPt8,deltaPt5) 0 ...
                funDN3Dksi(ksi,eta,deltaPt5,deltaPt6) 0 funDN4Dksi(ksi,eta,deltaPt6,deltaPt7) 0 ...
                funDN5Dksi(ksi,eta,deltaPt5) 0 funDN6Dksi(ksi,eta,deltaPt6) 0 ...
                funDN7Dksi(ksi,eta,deltaPt7) 0 funDN8Dksi(ksi,eta,deltaPt8) 0;
                funDN1Deta(ksi,eta,deltaPt7,deltaPt8) 0 funDN2Deta(ksi,eta,deltaPt8,deltaPt5) 0 ...
                funDN3Deta(ksi,eta,deltaPt5,deltaPt6) 0 funDN4Deta(ksi,eta,deltaPt6,deltaPt7) 0 ...
                funDN5Deta(ksi,eta,deltaPt5) 0 funDN6Deta(ksi,eta,deltaPt6) 0 ...
                funDN7Deta(ksi,eta,deltaPt7) 0 funDN8Deta(ksi,eta,deltaPt8) 0;
                0 funDN1Dksi(ksi,eta,deltaPt7,deltaPt8) 0 funDN2Dksi(ksi,eta,deltaPt8,deltaPt5) ...
                0 funDN3Dksi(ksi,eta,deltaPt5,deltaPt6) 0 funDN4Dksi(ksi,eta,deltaPt6,deltaPt7) ...
                0 funDN5Dksi(ksi,eta,deltaPt5) 0 funDN6Dksi(ksi,eta,deltaPt6) ...
                0 funDN7Dksi(ksi,eta,deltaPt7) 0 funDN8Dksi(ksi,eta,deltaPt8);
                0 funDN1Deta(ksi,eta,deltaPt7,deltaPt8) 0 funDN2Deta(ksi,eta,deltaPt8,deltaPt5) ...
                0 funDN3Deta(ksi,eta,deltaPt5,deltaPt6) 0 funDN4Deta(ksi,eta,deltaPt6,deltaPt7) ...
                0 funDN5Deta(ksi,eta,deltaPt5) 0 funDN6Deta(ksi,eta,deltaPt6) ...
                0 funDN7Deta(ksi,eta,deltaPt7) 0 funDN8Deta(ksi,eta,deltaPt8)];
            
            % ------- Comment: Calculate DivFMinusW ---------
            % tempDUDX = DN*FMinusW1(temp);
            % DivFMinusW1 = tempDUDX(1)+tempDUDX(4);
            % tempDUDX = DN*FMinusW2(temp);
            % DivFMinusW2 = tempDUDX(1)+tempDUDX(4);
            % DivFMinusW = [DivFMinusW1;DivFMinusW2;DivFMinusW1;DivFMinusW2;
            %               DivFMinusW1;DivFMinusW2;DivFMinusW1;DivFMinusW2;
            %               DivFMinusW1;DivFMinusW2;DivFMinusW1;DivFMinusW2;
            %               DivFMinusW1;DivFMinusW2;DivFMinusW1;DivFMinusW2];
            % ------ End of comment of calculating DivFMinusW ------
            
            % ------ Construct A matrix ------
            tempA = tempA + Jacobian*weightksi*weighteta * ( (beta+alpha)*(DN')*(DN) + mu*(N')*N  );
            
            % ------ Construct b vector ------
            tempb = tempb + Jacobian*weightksi*weighteta * ( beta*diag((DN')*(FMinusW(tempIndexF)')) + mu*N'*N*(UMinusv(tempIndexU)) +(alpha)*(DN')*DN*U(tempIndexU) );
            
            %%%%% %end %%%%%
        %%%%% %end %%%%%
        end
        
        
        % Store tempA and tempb
        if IterStep==1
            [IndexAXX,IndexAYY] = ndgrid(tempIndexU,tempIndexU);
            INDEXAIpar{eleInd}=IndexAXX(:); INDEXAJpar{eleInd}=IndexAYY(:); INDEXAVALpar{eleInd}=tempA(:);
        end
        INDEXBIpar{eleInd}=tempIndexU(:); INDEXBVALpar{eleInd}=tempb(:);   
        
        
    end
    
    if waitBarDisplayOrNot == 0
      close(hbar);
    end
    
    % ====== Initialize A matrix and b vector ======
    A = sparse(FEMSize+DIM*NodesNumPerEle, FEMSize+DIM*NodesNumPerEle);   b=sparse(FEMSize+DIM*NodesNumPerEle,1);
    for eleInd = 1:size(elementsFEM,1)
        A = A + sparse(INDEXAIpar{eleInd}, INDEXAJpar{eleInd}, INDEXAVALpar{eleInd}, FEMSize+DIM*NodesNumPerEle , FEMSize+DIM*NodesNumPerEle) ;
        b = b + sparse(INDEXBIpar{eleInd},ones(length(INDEXBIpar{eleInd}),1),INDEXBVALpar{eleInd}, FEMSize+DIM*NodesNumPerEle , 1) ;
    end
    % Adding "DIM*NodesNumPerEle" at the end of Matrix A and vector b 
    % is because I put all the zeros appearing in elementsFEM to the end 
 
    % ========= Adding regularization if needed =========
    %if IterStep == 1
        % AMatrixRegularized = A + 1e-3*max(diag(A))*speye(size(A,1),size(A,2));
        % AMatrixRegularized = A; tempVal = 1e-3*max(diag(A));
        % for tempi = 1:size(A,1)
        %     AMatrixRegularized(tempi,tempi) = AMatrixRegularized(tempi,tempi) + tempVal;
        % send
    %end
    
    % ========= Finding evolved nodal points =========
    coordsIndexInvolved = unique(elementsFEM(:,1:8)); % Need modification for triangle elementsFEM
    UIndexInvolved = zeros(2*(length(coordsIndexInvolved)-1),1); % Not to include the first 0-th entry
    for tempi = 1:(size(coordsIndexInvolved,1)-1)
        UIndexInvolved(2*tempi-1:2*tempi) = [2*coordsIndexInvolved(tempi+1)-1; 2*coordsIndexInvolved(tempi+1)];
    end
    
    % ========= Set Dirichlet and Neumann boundary conditions =========
    if isempty(dirichlet) ~= 1
        dirichlettemp = [2*dirichlet(:); 2*dirichlet(:)-1];
    else
        dirichlettemp = [];
    end
    % if isempty(neumann) ~= 1
    %     neumanntemp = [2*neumann(:,1); 2*neumann(:,1)-1; 2*neumann(:,2); 2*neumann(:,2)-1];
    % else
    %     neumanntemp = [];
    % end
    FreeNodes = setdiff(UIndexInvolved,unique([dirichlettemp]));
    
    % ========= Neumann conditions ===========
    % Last step boundary condition force
    BCForce = - 1/winstepsize * F;
    for tempj = 1:size(neumann,1)
        b(2*neumann(tempj,1:2)-1) = b(2*neumann(tempj,1:2)-1) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
          *( ( BCForce(4*neumann(tempj,1:2)-3) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)-1) * neumann(tempj,4) ) );
        b(2*neumann(tempj,1:2))   = b(2*neumann(tempj,1:2)) + 0.5*norm(coordinatesFEM(neumann(tempj,1),:)-coordinatesFEM(neumann(tempj,2),:)) ...
         * (( BCForce(4*neumann(tempj,1:2)-2) * neumann(tempj,3) + BCForce(4*neumann(tempj,1:2)) * neumann(tempj,4) ) );
    end
    
    % ========= Dirichlet conditions ==========
    UhatOld = Uhat; Uhat = sparse(FEMSize+DIM*NodesNumPerEle,1);
    for tempi = 1:DIM
        Uhat(DIM*unique(dirichlet)-(tempi-1)) = U(DIM*unique(dirichlet)-(tempi-1)); 
    end
    b = b - A * Uhat;
     
    % ========= Solve FEM problem ===========
    Uhat(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);
    UhatNew = Uhat(1:end-DIM*NodesNumPerEle);
     
    if norm(UhatNew-UhatOld)/sqrt(length(UhatOld)) < 1e-3
        ConvergeOrNot = 1;
    end
    
    Uhat = full(UhatNew);
    
    
    
end

 

end


%% ========= subroutines for  FEM Q4 shape function derivatives ========
function DN5Dksi=funDN5Dksi(ksi,eta,deltaPt5)
    DN5Dksi = deltaPt5*0.5*(1-abs(eta)) ;
    %DN5Dksi = deltaPt5*0.5*(1-eta^2); %%%%% Old code %%%%%
end
function DN5Deta=funDN5Deta(ksi,eta,deltaPt5)
    DN5Deta = deltaPt5*0.5*(1+ksi)*sign(-eta);
    %DN5Deta = deltaPt5*(-1)*(1+ksi)*eta; %%%%% Old code %%%%%
end
function DN6Dksi=funDN6Dksi(ksi,eta,deltaPt6)
    DN6Dksi = deltaPt6*0.5*(1+eta)*sign(-ksi);
    %DN6Dksi = deltaPt6*(-1)*(1+eta)*ksi; %%%%% Old code %%%%%
end
function DN6Deta=funDN6Deta(ksi,eta,deltaPt6)
    DN6Deta = deltaPt6*0.5*(1-abs(ksi));
    %DN6Deta = deltaPt6*0.5*(1-ksi^2); %%%%% Old code %%%%%
end
function DN7Dksi=funDN7Dksi(ksi,eta,deltaPt7)
    DN7Dksi = deltaPt7*0.5*(-1)*(1-abs(eta));
    %DN7Dksi = deltaPt7*(-0.5)*(1-eta^2); %%%%% Old code %%%%%
end
function DN7Deta=funDN7Deta(ksi,eta,deltaPt7)
    DN7Deta = deltaPt7*0.5*(1-ksi)*sign(-eta);
    %DN7Deta = deltaPt7*(-1)*(1-ksi)*eta; %%%%% Old code %%%%%
end
function DN8Dksi=funDN8Dksi(ksi,eta,deltaPt8)
    DN8Dksi = deltaPt8*0.5*(1-eta)*sign(-ksi);
    %DN8Dksi = deltaPt8*(-1)*(1-eta)*ksi; %%%%% Old code %%%%%
end
function DN8Deta=funDN8Deta(ksi,eta,deltaPt8)
    DN8Deta = deltaPt8*0.5*(-1)*(1-abs(ksi));
    %DN8Deta = deltaPt8*(-0.5)*(1-ksi^2); %%%%% Old code %%%%%
end
function DN1Dksi = funDN1Dksi(ksi,eta,deltaPt7,deltaPt8)
    DN1Dksi = -0.25*(1-eta)-0.5*( funDN7Dksi(ksi,eta,deltaPt7) + funDN8Dksi(ksi,eta,deltaPt8) );
end
function DN1Deta = funDN1Deta(ksi,eta,deltaPt7,deltaPt8)
    DN1Deta = -0.25*(1-ksi)-0.5*( funDN7Deta(ksi,eta,deltaPt7) + funDN8Deta(ksi,eta,deltaPt8) );
end
function DN2Dksi = funDN2Dksi(ksi,eta,deltaPt8,deltaPt5)
    DN2Dksi = 0.25*(1-eta)-0.5*( funDN8Dksi(ksi,eta,deltaPt8) + funDN5Dksi(ksi,eta,deltaPt5) );
end
function DN2Deta = funDN2Deta(ksi,eta,deltaPt8,deltaPt5)
    DN2Deta = -0.25*(1+ksi)-0.5*( funDN8Deta(ksi,eta,deltaPt8) + funDN5Deta(ksi,eta,deltaPt5) );
end
function DN3Dksi = funDN3Dksi(ksi,eta,deltaPt5,deltaPt6)
    DN3Dksi = 0.25*(1+eta)-0.5*( funDN5Dksi(ksi,eta,deltaPt5) + funDN6Dksi(ksi,eta,deltaPt6) );
end
function DN3Deta = funDN3Deta(ksi,eta,deltaPt5,deltaPt6)
    DN3Deta = 0.25*(1+ksi)-0.5*( funDN5Deta(ksi,eta,deltaPt5) + funDN6Deta(ksi,eta,deltaPt6) );
end
function DN4Dksi = funDN4Dksi(ksi,eta,deltaPt6,deltaPt7)
    DN4Dksi = -0.25*(1+eta)-0.5*( funDN6Dksi(ksi,eta,deltaPt6) + funDN7Dksi(ksi,eta,deltaPt7) );
end
function DN4Deta = funDN4Deta(ksi,eta,deltaPt6,deltaPt7)
    DN4Deta = 0.25*(1-ksi)-0.5*( funDN6Deta(ksi,eta,deltaPt6) + funDN7Deta(ksi,eta,deltaPt7) );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 


