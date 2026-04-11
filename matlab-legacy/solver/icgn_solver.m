function [U,F,stepwithinwhile] = icgn_solver(U0,x0,y0,Df,ImgRef,ImgDef,winsize,tol)
%FUNCTION [U,F,stepwithinwhile] = icgn_solver(U0,x0,y0,Df,ImgRef,ImgDef,winsize,tol)
% The Local ICGN subset solver: Gauss-Newton IC-GN iteration (6-DOF)
% (see dispatcher: ./solver/local_icgn.m)
% ----------------------------------------------
%   INPUT: U0                   Initial guess of the displacement fields
%          x0,y0                FE mesh nodal coordinates
%          Df                   Image grayscale value gradients
%          ImgRef               Reference image
%          ImgDef               Deformed image
%          winsize              DIC parameter subset size
%          tol                  ICGN iteration stopping threshold
%
%   OUTPUT: U                   Disp vector: [Ux_node1, Uy_node1, ... , Ux_nodeN, Uy_nodeN]';
%           F                   Deformation gradient tensor
%                               F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%           stepwithinwhile     ICGN iteration step for convergence
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================

%% Initialization
warnState = warning('off', 'MATLAB:nearlySingularMatrix');
cleanupWarn = onCleanup(@() warning(warnState));
DfCropWidth = Df.DfCropWidth;
imgSize = Df.imgSize;
winsize0 = winsize;


%% ---------------------------
% Find local subset region
x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2];  % [coordinates(elements(j,:),1)];
y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2];  % [coordinates(elements(j,:),2)];

% ---------------------------
% Initialization: Get P0
P0 = [0 0 0 0 U0(1) U0(2)]';
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

%% %%%%%%%% If there are >60% of the subset are painted with patterns %%%%%%%%%%%%
maxIterNum = 100;
%%%%%%%%%%%% !!!Mask START %%%%%%%%%%%%%%
DfDxImgMaskIndCount = sum(double(1-logical(tempf(:))));
% [DfDxImgMaskIndRow,~] = find(abs(tempf)<1e-10);
% DfDxImgMaskIndCount = length(DfDxImgMaskIndRow);
%%%%%%%%%%%% !!!Mask END %%%%%%%%%%%%%%

Threshold = 0.4; % before is 0.4
if DfDxImgMaskIndCount <= Threshold*(winsize+1)^2
     
    if DfDxImgMaskIndCount > 0.0*(winsize+1)^2 % For those subsets where are 0's in the image mask file
         
        winsize = 2*max(ceil(sqrt((winsize+1)^2/((winsize+1)^2-DfDxImgMaskIndCount))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels
        x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2]; % Update x
        y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2]; % Update y
        [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]); 
        tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
        tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
        DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
        DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
         
    end
    
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    %%%%% Find connected region to deal with possible continuities %%%%%
    tempf_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 );
    DfDx_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 );
    DfDy_BW2 = bwselect(logical(tempfImgMask), floor((winsize+1)/2), floor((winsize+1)/2), 4 );
    tempf = tempf .* double(tempf_BW2);
    DfDx = DfDx .* double(tempf_BW2);
    DfDy = DfDy .* double(tempf_BW2);
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    H2 = zeros(6,6); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
    XXSq = (XX-x0).^2; YYSq = (YY-y0).^2; XXYY = (XX-x0).*(YY-y0);
    H2(1,1) = sum(sum(XXSq.*DfDxSq));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy ));
    H2(1,3) = sum(sum( XXYY.*DfDxSq ));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy ));
    H2(1,5) = sum(sum( (XX-x0).*DfDxSq ));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy ));
    H2(2,2) = sum(sum(XXSq.*DfDySq));       H2(2,3) = H2(1,4);
    H2(2,4) = sum(sum( XXYY.*DfDySq ));     H2(2,5) = H2(1,6);
    H2(2,6) = sum(sum( (XX-x0).*DfDySq ));  H2(3,3) = sum(sum( YYSq.*DfDxSq ));
    H2(3,4) = sum(sum( YYSq.*DfDxDfDy ));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq ));
    H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy ));H2(4,4) = sum(sum( YYSq.*DfDySq ));
    H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq)); H2(5,5) = sum(sum(DfDxSq));
    H2(5,6) = sum(sum(DfDxDfDy)); H2(6,6) = sum(sum(DfDySq));
    H = H2 + H2' - diag(diag(H2));
    
    %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
    meanf = mean(tempf(abs(tempf)>1e-10));
    bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
    %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    % --------------------------
    % Initialize while loop (Gauss-Newton: delta=0)
    normOfWOld=2; normOfWNew=1; normOfWNewAbs=1; stepwithinwhile=0;
    delta = 0;
    
    while( (stepwithinwhile <= maxIterNum) && (normOfWNew>tol) && (normOfWNewAbs>tol) )
        
        stepwithinwhile = stepwithinwhile+1;
        
        if stepwithinwhile>1 && DfDxImgMaskIndCount>0.0*(winsize0+1)^2
            winsize = 2*max(ceil(sqrt((winsize0+1)^2/(sum(double(tempg_BW2(:)))))*winsize/2)); % Increase the subset size a bit to guarantuee there there enough pixels
            x = [x0-winsize/2 ; x0+winsize/2 ; x0+winsize/2 ; x0-winsize/2]; % Update x
            y = [y0-winsize/2 ; y0+winsize/2 ; y0+winsize/2 ; y0-winsize/2]; % Update y
            [XX,YY] = ndgrid([x(1):1:x(3)],[y(1):1:y(3)]);
            tempfImgMask = Df.ImgRefMask([x(1):1:x(3)],[y(1):1:y(3)]);
        end
        
        % Find region for g
        % %[tempCoordy, tempCoordx] = meshgrid(y(1):y(3),x(1):x(3));
        tempCoordxMat = XX - x0*ones(winsize+1,winsize+1);
        tempCoordyMat = YY - y0*ones(winsize+1,winsize+1);
        u22 = (1+P(1))*tempCoordxMat + P(3)*tempCoordyMat + (x0+P(5))*ones(winsize+1,winsize+1);
        v22 = P(2)*tempCoordxMat + (1+P(4))*tempCoordyMat + (y0+P(6))*ones(winsize+1,winsize+1);
        
        row1 = find(u22<3); row2 = find(u22>imgSize(1)-2); row3 = find(v22<3); row4 = find(v22>imgSize(2)-2);
        if ~isempty([row1; row2; row3; row4])
            normOfWNew = 1e6; % warning('Out of image boundary!')
            break;
        else
            
            %tempg = imggNormalizedbc.eval(u22,v22)
            %tempg = ba_interp2(ImgDef, v22, u22, 'cubic');

            % BicubicBspline interpolation % Zach 20250716
            tempg = ba_interp2_spline(ImgDef, v22, u22, 'cubicspline');
            
            DgDxImgMaskIndCount = sum(double(1-logical(tempg(:))));
            
            
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            if DfDxImgMaskIndCount>0.0*(winsize0+1)^2 || DgDxImgMaskIndCount>0.0*(winsize0+1)^2
                  
                %%%%% Find connected region to deal with possible continuities %%%%%
                % tempg_BW2 = logical(tempg);
                tempg_BW2 = bwselect(logical(tempg), floor((winsize+1)/2), floor((winsize+1)/2), 8 );
                
                [rowtemp,~] = find(tempg_BW2==0);
                if isempty(rowtemp)
                    tempg_BW2 = tempfImgMask;
                    tempg_BW2 = bwselect(tempg_BW2, floor((winsize+1)/2), floor((winsize+1)/2), 8 );
                end
                tempg = tempg .* double(tempg_BW2);

                tempf = ImgRef([x(1):1:x(3)],[y(1):1:y(3)]) .* tempfImgMask;
                tempf = tempf .* double(tempg_BW2);

                DfDx = Df.DfDx((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
                DfDy = Df.DfDy((x(1)-DfCropWidth):1:(x(3)-DfCropWidth), (y(1)-DfCropWidth):1:(y(3)-DfCropWidth));
                
                DfDx = DfDx .* double(tempg_BW2);
                DfDy = DfDy .* double(tempg_BW2);
                 
                H2 = zeros(6,6); DfDxSq = (DfDx.^2); DfDySq = (DfDy.^2); DfDxDfDy = DfDx.*DfDy;
                XXSq = (XX-x0).^2; YYSq = (YY-y0).^2; XXYY = (XX-x0).*(YY-y0);
                H2(1,1) = sum(sum(XXSq.*DfDxSq));       H2(1,2) = sum(sum(XXSq.*DfDxDfDy ));
                H2(1,3) = sum(sum( XXYY.*DfDxSq ));     H2(1,4) = sum(sum( XXYY.*DfDxDfDy ));
                H2(1,5) = sum(sum( (XX-x0).*DfDxSq ));  H2(1,6) = sum(sum( (XX-x0).*DfDxDfDy ));
                H2(2,2) = sum(sum(XXSq.*DfDySq));       H2(2,3) = H2(1,4);
                H2(2,4) = sum(sum( XXYY.*DfDySq ));     H2(2,5) = H2(1,6);
                H2(2,6) = sum(sum( (XX-x0).*DfDySq ));  H2(3,3) = sum(sum( YYSq.*DfDxSq ));
                H2(3,4) = sum(sum( YYSq.*DfDxDfDy ));   H2(3,5) = sum(sum( (YY-y0).*DfDxSq ));
                H2(3,6) = sum(sum( (YY-y0).*DfDxDfDy ));H2(4,4) = sum(sum( YYSq.*DfDySq ));
                H2(4,5) = H2(3,6);  H2(4,6) = sum(sum((YY-y0).*DfDySq)); H2(5,5) = sum(sum(DfDxSq));
                H2(5,6) = sum(sum(DfDxDfDy)); H2(6,6) = sum(sum(DfDySq));
                H = H2 + H2' - diag(diag(H2));

                %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
                meanf = mean(tempf(abs(tempf)>1e-10));
                bottomf = sqrt((length(tempf(abs(tempf)>1e-10))-1)*var(tempf(abs(tempf)>1e-10)));
                %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%


            end
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
    
    
            %%%%%%%%%% !!!Mask: START %%%%%%%%%%%%
            %%%%% Find connected region to deal with possible continuities %%%%%
            meang = mean(tempg(abs(tempg)>1e-10));
            bottomg = sqrt((length(tempg(abs(tempg)>1e-10))-1)*var(tempg(abs(tempg)>1e-10)));
            %%%%%%%%%% !!!Mask: END %%%%%%%%%%%%
            
            % ====== Assemble b vector (vectorized) ======
            b2 = zeros(6,1);
            tempfMinustempg = (tempf-meanf*ones(winsize+1,winsize+1))/bottomf - (tempg-meang*ones(winsize+1,winsize+1))/bottomg;
            b2(1) = sum(sum( (XX-x0).*DfDx.*tempfMinustempg ));
            b2(2) = sum(sum( (XX-x0).*DfDy.*tempfMinustempg ));
            b2(3) = sum(sum( (YY-y0).*DfDx.*tempfMinustempg ));
            b2(4) = sum(sum( (YY-y0).*DfDy.*tempfMinustempg ));
            b2(5) = sum(sum( DfDx.*tempfMinustempg ));
            b2(6) = sum(sum( DfDy.*tempfMinustempg ));
            
            b = bottomf * b2;
            
            normOfWOld = normOfWNew;
            normOfWNew = norm(b(:)); normOfWNewAbs = normOfWNew;
            
            if stepwithinwhile ==1
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
                % DeltaP = [0 0 0 0 0 0];
                % tempH = (H + delta*diag(diag(H)));
                % tempb = b;
                % DeltaP(5:6) = -tempH(5:6,5:6)\tempb(5:6);
                DeltaP = -(H + delta*diag(diag(H))) \ b;
                P = icgn_compose_warp(P, DeltaP);
                if isempty(P)
                    disp('Det(DeltaP)==0!')
                    break
                end
                
            end
        end
    end % end of while
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    if (normOfWNew<tol) || (normOfWNewAbs<tol)
        % elementsLocalMethodConvergeOrNot = 1;
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
else % if norm(diag(H)) > abs(eps)
    
    H = zeros(6,6);
    stepwithinwhile = maxIterNum+2;


end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U(1) = P(5); U(2) = P(6);
F(1) = P(1); F(2) = P(2); F(3) = P(3); F(4) = P(4);
 

end

