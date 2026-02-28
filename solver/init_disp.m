function U0 =init_disp(u,v,Phi,x,y,index)
%FUNCTION U0 =init_disp(u,v,Phi,x,y,index)
% ----------------------------------------------
% Remove outliers and smooth initial displacements

% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 02/2020.
% ==============================================
 
%% inpaint_nans: I use a Spring Model for u and v initial guesses
u = inpaint_nans(u,4); v = inpaint_nans(v,4);
uInit = u; vInit = v; 
% threshod = 0.5;
% [row, col] = find(Phi>2*(1-threshod));
% uInit(row,col) = NaN; vInit(row,col) = NaN;
% 
% uInit = inpaint_nans(uInit,4);
% vInit = inpaint_nans(vInit,4);

% %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
% figure;surf(uInit)
% colorbar
% figure;surf(vInit)
% colorbar
% %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

if (index == 1)

    utemp = uInit; vtemp = vInit;
    Mtemp = size(u,2); Ntemp = size(u,1);
    for i = 2:Ntemp-1
        for j = 2:Mtemp-1
            NeighborOfuInit = [utemp(i-1,j); utemp(i+1,j); utemp(i,j-1); utemp(i,j+1);
                utemp(i-1,j-1); utemp(i+1,j-1); utemp(i+1,j+1); utemp(i-1,j+1)];
            avgOfuInit(i,j) = (1/8*sum(NeighborOfuInit(1:4)) + 1/16*sum(NeighborOfuInit(5:8)))/(3/4);
              stdOfuInit(i,j) = std(NeighborOfuInit,'omitnan');

            NeighborOfvInit = [vtemp(i-1,j); vtemp(i+1,j); vtemp(i,j-1); vtemp(i,j+1);
                vtemp(i-1,j-1); vtemp(i+1,j-1); vtemp(i+1,j+1); vtemp(i-1,j+1)];
            avgOfvInit(i,j) = (1/8*sum(NeighborOfvInit(1:4)) + 1/16*sum(NeighborOfvInit(5:8)))/(3/4);
            stdOfvInit(i,j) = std(NeighborOfvInit,'omitnan');

            springErrOfu(i,j) = abs(utemp(i,j)-avgOfuInit(i,j)) - 3*stdOfuInit(i,j);          
            springErrOfv(i,j) = abs(vtemp(i,j)-avgOfvInit(i,j)) - 3*stdOfvInit(i,j); 

            if ( (springErrOfu(i,j)>0) || (springErrOfv(i,j)>0) )
                uInit(i,j) = NaN; vInit(i,j) = NaN; 
                uInit(i-1,j) = NaN; uInit(i+1,j) = NaN; uInit(i,j-1) = NaN; uInit(i,j+1) = NaN;
                uInit(i-1,j-1) = NaN; uInit(i+1,j-1) = NaN; uInit(i+1,j+1) = NaN; uInit(i-1,j+1) = NaN;
                vInit(i-1,j) = NaN; vInit(i+1,j) = NaN; vInit(i,j-1) = NaN; vInit(i,j+1) = NaN; 
                vInit(i-1,j-1) = NaN; vInit(i+1,j-1) = NaN; vInit(i+1,j+1) = NaN; vInit(i-1,j+1) = NaN;
            end
        end
    end

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure; surf(springErrOfu)
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    uInit(1,:) = NaN; uInit(end,:) = NaN; uInit(:,1) = NaN; uInit(:,end) = NaN;
    vInit(1,:) = NaN; vInit(end,:) = NaN; vInit(:,1) = NaN; vInit(:,end) = NaN;

    uInit = inpaint_nans(uInit,4);
    vInit = inpaint_nans(vInit,4);

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure;surf(uInit)
    % colorbar
    % figure;surf(vInit)
    % colorbar
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    % Second time
    utemp = uInit; vtemp = vInit;
    Mtemp = size(u,2); Ntemp = size(u,1);
    for i = 2:Ntemp-1
        for j = 2:Mtemp-1
            NeighborOfuInit = [utemp(i-1,j); utemp(i+1,j); utemp(i,j-1); utemp(i,j+1)];
            avgOfuInit(i,j) = mean(NeighborOfuInit,'omitnan');
              stdOfuInit(i,j) = std(NeighborOfuInit,'omitnan');

            NeighborOfvInit = [vtemp(i-1,j); vtemp(i+1,j); vtemp(i,j-1); vtemp(i,j+1)];
            avgOfvInit(i,j) = mean(NeighborOfvInit,'omitnan');
            stdOfvInit(i,j) = std(NeighborOfvInit,'omitnan');

            springErrOfu(i,j) = abs(utemp(i,j)-avgOfuInit(i,j)) - 2*stdOfuInit(i,j);          
            springErrOfv(i,j) = abs(vtemp(i,j)-avgOfvInit(i,j)) - 2*stdOfvInit(i,j); 

            if ( (springErrOfu(i,j)>0) || (springErrOfv(i,j)>0) )
                uInit(i,j) = NaN; vInit(i,j) = NaN; 
                uInit(i-1,j) = NaN; uInit(i+1,j) = NaN; uInit(i,j-1) = NaN; uInit(i,j+1) = NaN;
                % uInit(i-1,j-1) = NaN; uInit(i+1,j-1) = NaN; uInit(i+1,j+1) = NaN; uInit(i-1,j+1) = NaN;
                vInit(i-1,j) = NaN; vInit(i+1,j) = NaN; vInit(i,j-1) = NaN; vInit(i,j+1) = NaN; 
                % vInit(i-1,j-1) = NaN; vInit(i+1,j-1) = NaN; vInit(i+1,j+1) = NaN; vInit(i-1,j+1) = NaN;
            end
        end
    end

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure; surf(springErrOfu)
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    uInit(1,:) = NaN; uInit(end,:) = NaN; uInit(:,1) = NaN; uInit(:,end) = NaN;
    vInit(1,:) = NaN; vInit(end,:) = NaN; vInit(:,1) = NaN; vInit(:,end) = NaN;

    uInit = inpaint_nans(uInit,4);
    vInit = inpaint_nans(vInit,4);

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure;surf(uInit)
    % colorbar
    % figure;surf(vInit)
    % colorbar
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    % Third time
    utemp = uInit; vtemp = vInit;
    Mtemp = size(u,2); Ntemp = size(u,1);
    for i = 2:Ntemp-1
        for j = 2:Mtemp-1
            NeighborOfuInit = [utemp(i-1,j); utemp(i+1,j); utemp(i,j-1); utemp(i,j+1);
                utemp(i-1,j-1); utemp(i+1,j-1); utemp(i+1,j+1); utemp(i-1,j+1)];
            avgOfuInit(i,j) = (1/8*sum(NeighborOfuInit(1:4)) + 1/16*sum(NeighborOfuInit(5:8)))/(3/4);
              stdOfuInit(i,j) = std(NeighborOfuInit,'omitnan');

            NeighborOfvInit = [vtemp(i-1,j); vtemp(i+1,j); vtemp(i,j-1); vtemp(i,j+1);
                vtemp(i-1,j-1); vtemp(i+1,j-1); vtemp(i+1,j+1); vtemp(i-1,j+1)];
            avgOfvInit(i,j) = (1/8*sum(NeighborOfvInit(1:4)) + 1/16*sum(NeighborOfvInit(5:8)))/(3/4);
            stdOfvInit(i,j) = std(NeighborOfvInit,'omitnan');

            springErrOfu(i,j) = abs(utemp(i,j)-avgOfuInit(i,j)) - 2*stdOfuInit(i,j);          
            springErrOfv(i,j) = abs(vtemp(i,j)-avgOfvInit(i,j)) - 2*stdOfvInit(i,j); 

            if ( (springErrOfu(i,j)>0) || (springErrOfv(i,j)>0) )
                uInit(i,j) = NaN; vInit(i,j) = NaN; 
                uInit(i-1,j) = NaN; uInit(i+1,j) = NaN; uInit(i,j-1) = NaN; uInit(i,j+1) = NaN;
                uInit(i-1,j-1) = NaN; uInit(i+1,j-1) = NaN; uInit(i+1,j+1) = NaN; uInit(i-1,j+1) = NaN;
                vInit(i-1,j) = NaN; vInit(i+1,j) = NaN; vInit(i,j-1) = NaN; vInit(i,j+1) = NaN; 
                vInit(i-1,j-1) = NaN; vInit(i+1,j-1) = NaN; vInit(i+1,j+1) = NaN; vInit(i-1,j+1) = NaN;
            end
        end
    end

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure; surf(springErrOfu)
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    uInit(1,:) = NaN; uInit(end,:) = NaN; uInit(:,1) = NaN; uInit(:,end) = NaN;
    vInit(1,:) = NaN; vInit(end,:) = NaN; vInit(:,1) = NaN; vInit(:,end) = NaN;

    uInit = inpaint_nans(uInit,4);
    vInit = inpaint_nans(vInit,4);

    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%
    % figure;surf(uInit)
    % colorbar
    % figure;surf(vInit)
    % colorbar
    % %%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%

    clear Mtemp Ntemp 
    clear NeighborOfuInit NeighborOfvInit avgOfuInit avgOfvInit stdOfuInit stdOfvInit springErrOfu springErrOfv

end

%% Initial Value
%  U000 = zeros(2*size(x,1)*size(y,2),1); % Phi0 = zeros(size(x,1)*size(y,2),1);
% uInit = uInit'; vInit = vInit'; % Here transpose bcz following give valus in column order
% % PhiInit = Phi';
% for i = 1:(size(x,1)*size(y,2))
%     U000(2*i-1) = uInit(i); % 0; %u(i);
%     U000(2*i)   = vInit(i); % 0; %v(i);
%     % Phi0(i)   = PhiInit(i); 
% end
% U0 = U000;

% Zach modified
uInit = uInit'; vInit = vInit'; % Here transpose bcz following give valus in column order
n = numel(uInit);
U0  = zeros(2*n, 1);
U0 (1:2:end) = uInit;
U0 (2:2:end) = vInit; 

disp('Finish setting up mesh and assigning initial value!')
 



