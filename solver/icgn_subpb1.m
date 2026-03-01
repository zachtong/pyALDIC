function [U,stepwithinwhile] = icgn_subpb1(x0,y0,Df,ImgRef,ImgDef,winsize_x,winsize_y,...
                                                beta,mu,udual,vdual,UOld,FOld,tol)
%FUNCTION [U,stepwithinwhile] = icgn_subpb1(x0,y0,Df,ImgRef,ImgDef,winsize_x,winsize_y,...
%                                                beta,mu,udual,vdual,UOld,FOld,tol)
% The ALDIC Subproblem 1 ICGN subset solver: Gauss-Newton IC-GN iteration (2-DOF)
% (see dispatcher: ./solver/subpb1_solver.m)
% ----------------------------------------------
%   INPUT: x0,y0                FE mesh nodal coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          winsize_x,winsize_y  DIC parameter subset size (may differ per axis)
%          beta,mu              ALDIC coefficients
%          udual,vdual          Dual variables
%          UOld                 Initial guess of the displacement fields
%          FOld                 Initial guess of the deformation gradients
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           stepwithinwhile     ICGN iteration step for convergence
%            
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 2018.03, 2020.12.
% ==============================================
                                            
                                           
%% Initialization
warnState = warning('off', 'MATLAB:nearlySingularMatrix');
cleanupWarn = onCleanup(@() warning(warnState));
DfCropWidth = Df.DfCropWidth;
imgSize = Df.imgSize;

% winsize_x = 2*ceil( max([  min([ winsize, abs(UOld(1)/FOld(1)), abs(UOld(2)/FOld(2)) ])/2, 5]) );
% winsize_y = 2*ceil( max([  min([ winsize, abs(UOld(1)/FOld(3)), abs(UOld(2)/FOld(4)) ])/2, 5]) );

%% ---------------------------
% Find local subset region
x  = [x0-winsize_x/2 ; x0+winsize_x/2 ; x0+winsize_x/2 ; x0-winsize_x/2];  
y  = [y0-winsize_y/2 ; y0-winsize_y/2 ; y0+winsize_y/2 ; y0+winsize_y/2];   

% ---------------------------
% Initialization: Get P0
P0 = [FOld(1) FOld(2) FOld(3) FOld(4) UOld(1) UOld(2)]'; 
P = P0;

% ---------------------------
% Find region for f
[XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]);
%tempf = imgfNormalizedbc.eval(XX,YY); 
%DfDx = imgfNormalizedbc.eval_Dx(XX,YY);
%DfDy = imgfNormalizedbc.eval_Dy(XX,YY); 

%%%%%%%%%%%% !!!Mask START %%%%%%%%%%%%%%
tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
%%%%%%%%%%%% !!!Mask END %%%%%%%%%%%%%%
DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));


%% %%%%%%%% If there are >50% of the subset are painted with patterns %%%%%%%%%%%%
maxIterNum = 100;
%%%%%%%%%%%% !!!Mask START %%%%%%%%%%%%%%
DfDxImgMaskIndCount = sum(double(1-logical(tempf(:))));
% [DfDxImgMaskIndRow,~] = find(abs(tempf)<1e-10);
% DfDxImgMaskIndCount = length(DfDxImgMaskIndRow);
%%%%%%%%%%%% !!!Mask END %%%%%%%%%%%%%%

if DfDxImgMaskIndCount < 0.4*(winsize_x+1)*(winsize_y+1)
    
    if DfDxImgMaskIndCount > 0.0*(winsize_x+1)*(winsize_y+1)
        
        % Increase the subset size a bit to guarantuee there there enough pixels
        winsize_x = 2*max(ceil(sqrt((winsize_x+1)*(winsize_y+1)/((winsize_x+1)*(winsize_y+1)-DfDxImgMaskIndCount))*winsize_x/2)); 
        winsize_y = 2*max(ceil(sqrt((winsize_x+1)*(winsize_y+1)/((winsize_x+1)*(winsize_y+1)-DfDxImgMaskIndCount))*winsize_y/2));
         
        x = [x0-winsize_x/2 ; x0+winsize_x/2 ; x0+winsize_x/2 ; x0-winsize_x/2]; % Update x
        y = [y0-winsize_y/2 ; y0+winsize_y/2 ; y0+winsize_y/2 ; y0-winsize_y/2]; % Update y
        [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]); 
        tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
        tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
        DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
        DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
    
    end
    
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    %%%%% Find connected region to deal with possible continuities %%%%%
    tempf_BW2 = bwselect(logical(tempfImgMask), floor((winsize_x+1)/2), floor((winsize_y+1)/2), 4 );
    DfDx_BW2 = bwselect(logical(tempfImgMask), floor((winsize_x+1)/2), floor((winsize_y+1)/2), 4 );
    DfDy_BW2 = bwselect(logical(tempfImgMask), floor((winsize_x+1)/2), floor((winsize_y+1)/2), 4 );
    tempf = tempf .* double(tempf_BW2);
    DfDx = DfDx .* double(tempf_BW2);
    DfDy = DfDy .* double(tempf_BW2);
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    H2 = zeros(2,2); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
    H2(1,1) = sum(sum(DfDxSq));
    H2(1,2) = sum(sum(DfDxDfDy)); H2(2,2) = sum(sum(DfDySq));
    H = H2 + H2' - diag(diag(H2));
    
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    meanf = mean(tempf(abs(tempf)>1e-10));
    bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    H2 = H*2/(bottomf^2) + [mu 0; 0 mu];
     
    % --------------------------
    % Initialize while loop (Gauss-Newton: delta=0)
    normOfWOld=2; normOfWNew=1; normOfWNewAbs=1; stepwithinwhile=0;
    delta = 0;
    
    
    while( (stepwithinwhile<=100) && (normOfWNew>tol) && (normOfWNewAbs>tol) )
        
        stepwithinwhile = stepwithinwhile+1;
        
        % Find region for g
        tempCoordxMat = XX - x0*ones(winsize_x+1,winsize_y+1);
        tempCoordyMat = YY - y0*ones(winsize_x+1,winsize_y+1);
        u22 = (1+P(1))*tempCoordxMat + P(3)*tempCoordyMat + (x0+P(5))*ones(winsize_x+1,winsize_y+1);
        v22 = P(2)*tempCoordxMat + (1+P(4))*tempCoordyMat + (y0+P(6))*ones(winsize_x+1,winsize_y+1);
        
        row1 = find(u22<3); row2 = find(u22>imgSize(1)-2); row3 = find(v22<3); row4 = find(v22>imgSize(2)-2);
        if ~isempty([row1; row2; row3; row4])
            normOfWNew = 1e6;
            % warning('Out of image boundary!')
            break;
        else
            
            %tempg = imggNormalizedbc.eval(u22,v22)
            tempg = ba_interp2_spline(ImgDef, v22, u22, 'cubicspline');
            
            
            DgDxImgMaskIndCount = sum(double(1-logical(tempg(:))));
            
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            if DfDxImgMaskIndCount>0.0*(winsize_x+1)*(winsize_y+1) || DgDxImgMaskIndCount>0.0*(winsize_x+1)*(winsize_y+1)
                
                %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                %%%%% Find connected region to deal with possible continuities %%%%%
                tempg_BW2 = bwselect(logical(tempg), floor((winsize_x+1)/2), floor((winsize_y+1)/2), 4 );
               
                [rowtemp,~] = find(tempg_BW2==0);
                if isempty(rowtemp)
                    tempg_BW2 = tempfImgMask;
                    tempg_BW2 = bwselect(tempg_BW2, floor((winsize_x+1)/2), floor((winsize_y+1)/2), 8 );
                end
                
                tempg = tempg .* double(tempg_BW2);


                tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
                tempf = tempf .* double(tempg_BW2);

                DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
                DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));

                DfDx = DfDx .* double(tempg_BW2);
                DfDy = DfDy .* double(tempg_BW2);
                
                H2 = zeros(2,2); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
                H2(1,1) = sum(sum(DfDxSq));
                H2(1,2) = sum(sum(DfDxDfDy)); H2(2,2) = sum(sum(DfDySq));
                H = H2 + H2' - diag(diag(H2));

                %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                meanf = mean(tempf(abs(tempf)>1e-10));
                bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
                %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%

                H2 = H*2/(bottomf^2) + [mu 0; 0 mu];



            end
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
            
            
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            %%%%% Find connected region to deal with possible continuities %%%%%
            meang = mean(tempg(abs(tempg)>1e-10));
            bottomg = sqrt((length(tempg(abs(tempg)>1e-10))-1)*var(tempg(abs(tempg)>1e-10)));
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
            
            % Assemble b vector (vectorized, 2-DOF only)
            b2 = zeros(2,1);
            tempfMinustempg = (tempf-meanf*ones(winsize_x+1,winsize_y+1))/bottomf - ...
                (tempg-meang*ones(winsize_x+1,winsize_y+1))/bottomg;
            % b2(1) = sum(sum( (XX-x0).*DfDx.*tempfMinustempg ));
            % b2(2) = sum(sum( (XX-x0).*DfDy.*tempfMinustempg ));
            % b2(3) = sum(sum( (YY-y0).*DfDx.*tempfMinustempg ));
            % b2(4) = sum(sum( (YY-y0).*DfDy.*tempfMinustempg ));
            b2(1) = sum(sum( DfDx.*tempfMinustempg ));
            b2(2) = sum(sum( DfDy.*tempfMinustempg ));
            
            b = bottomf * b2;
            
            tempb = b(1:2)*2/(bottomf^2)  + [mu*(P(5)-UOld(1)-vdual(1)); mu*(P(6)-UOld(2)-vdual(2))];
            
            
            delta = 1e-3; DeltaP = [0 0 0 0 0 0];
            tempH = (H2 + delta*max(diag(H2))*eye(2));
            DeltaP(5:6) = -tempH\tempb;
            
            normOfWOld = normOfWNew;
            normOfWNew = norm(tempb(:)); normOfWNewAbs = normOfWNew;
            
            if stepwithinwhile == 1
                normOfWNewInit = normOfWNew;
            end
            
            if normOfWNewInit > tol
                normOfWNew = normOfWNew/normOfWNewInit;
            else
                normOfWNew = 0;
            end
            
            if (normOfWNew<tol) || (normOfWNewAbs<tol)
                break
            else
                
                tempDelP =  ((1+DeltaP(1))*(1+DeltaP(4)) - DeltaP(2)*DeltaP(3));
                if (tempDelP ~= 0)
                    tempP1 =  (-DeltaP(1)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/tempDelP;
                    tempP2 =  -DeltaP(2)/tempDelP;
                    tempP3 =  -DeltaP(3)/tempDelP;
                    tempP4 =  (-DeltaP(4)-DeltaP(1)*DeltaP(4)+DeltaP(2)*DeltaP(3))/tempDelP;
                    tempP5 =  (-DeltaP(5)-DeltaP(4)*DeltaP(5)+DeltaP(3)*DeltaP(6))/tempDelP;
                    tempP6 =  (-DeltaP(6)-DeltaP(1)*DeltaP(6)+DeltaP(2)*DeltaP(5))/tempDelP;
                    
                    tempMatrix = [1+P(1) P(3) P(5); P(2) 1+P(4) P(6); 0 0 1]*...
                        [1+tempP1 tempP3 tempP5; tempP2 1+tempP4 tempP6; 0 0 1];
                    
                    P1 = tempMatrix(1,1)-1;
                    P2 = tempMatrix(2,1);
                    P3 = tempMatrix(1,2);
                    P4 = tempMatrix(2,2)-1;
                    P5 = tempMatrix(1,3);
                    P6 = tempMatrix(2,3);
                    P = [P1 P2 P3 P4 P5 P6]';
                else
                    disp( 'Det(DeltaP)==0!' );
                    break
                end
                
            end
        end
    end % end of while
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if (normOfWNew<tol) || (normOfWNewAbs<tol)
    else
        stepwithinwhile = maxIterNum+1;
    end
    
    if (isnan(normOfWNew)==1)
        stepwithinwhile = maxIterNum+1;
    end
    if sum(abs(tempf(:))) < 1e-6
        stepwithinwhile = maxIterNum+3;
    end
    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else

   stepwithinwhile = 102;

end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    U = [P(5);P(6)];
    % F(1) = P(1); F(2) = P(2); F(3) = P(3); F(4) = P(4);
    
end


