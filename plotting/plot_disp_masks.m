function plot_disp_masks(U,coordinatesFEMWorld,elementsFEM,CurrentImg,CurrentImgMask,DICpara)
%plot_disp_masks: to plot DIC solved displacement components
%   plot_disp_masks(U,coordinatesFEMWorld,elementsFEM,CurrentImg,DICpara)
% ----------------------------------------------
%
%   INPUT: U                    Displacement vector: 
%                               U = [Ux_node1, Uy_node1, Ux_node2, Uy_node2, ... , Ux_nodeN, Uy_nodeN]';
%          coordinatesFEM       FE mesh coordinates
%          elementsFEM          FE mesh elements
%          CurrentImg           Current deformed image
%          DICpara              DIC paramters
%
%   OUTPUT: Plots of x-displacement field and y-displacement field.
%
% ----------------------------------------------
% Reference
% [1] RegularizeNd. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/61436-regularizend
% [2] Gridfit. Matlab File Exchange open source. 
% https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last date modified: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
warning off; load('./plotting/colormap_RdYlBu.mat','cMap');
% run('./plotting/Black_rainbow.m'); % Black_rainbow.m removed in cleanup

% Zach edited
Img_temp = imread(CurrentImg);
if size(Img_temp,3) == 4
    Img_temp = Img_temp(:,:,1:3);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% convert pixel unit to the physical world unit %%%%%
try um2px = DICpara.um2px; 
catch um2px = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OrigDICImgTransparency = DICpara.OrigDICImgTransparency; % Original raw DIC image transparency
Image2PlotResults = DICpara.Image2PlotResults; % Choose image to plot over (first only, second and next images)

disp_u = U(1:2:end); disp_v = U(2:2:end);
coordinatesFEMWorldDef = [coordinatesFEMWorld(:,1)+Image2PlotResults*disp_u, coordinatesFEMWorld(:,2)+Image2PlotResults*disp_v];

%%%%%%%%%%% JY!!!Mask START %%%%%%%%%%%%%%%
if Image2PlotResults == 1
    if ~isempty(CurrentImgMask)
        for tempi = 1:size(coordinatesFEMWorldDef,1)
            try
                if CurrentImgMask( round(coordinatesFEMWorldDef(tempi,1)/um2px), ...
                                    (size(CurrentImgMask,2)+1-round(coordinatesFEMWorldDef(tempi,2)/um2px)) ) == 0 
                    coordinatesFEMWorldDef(tempi,:) = [nan,nan];
                end
            catch
                coordinatesFEMWorldDef(tempi,:) = [nan,nan];
            end

        end
    else
        CurrentImgMask = Img_temp';
        for tempi = 1:size(coordinatesFEMWorldDef,1)
            try
                if CurrentImgMask( round(coordinatesFEMWorldDef(tempi,1)/um2px), ...
                        (size(CurrentImgMask,2)+1-round(coordinatesFEMWorldDef(tempi,2)/um2px)) ) < 0
                    coordinatesFEMWorldDef(tempi,:) = [nan,nan];
                end
            catch
                coordinatesFEMWorldDef(tempi,:) = [nan,nan];
            end
        end
    end
end
%%%%%%%%%%% JY!!!Mask END %%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 1) dispx u ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes;  
try h1=imshow( flipud(Img_temp) ,'InitialMagnification','fit');
catch h1=surf(  flipud( Img_temp),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,disp_u,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency);  colormap(turbo); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(cMap); caxis([-40,40]); % colormap(turbo);  
% caxis([-35,35]); % caxis([-0.025,0.025]); 
% caxis([-1.1,1.1]);
% clim([-1.9,1.9]) % Indentation
% clim([ -23 , 15 ]) % Bio-printing
% colormap(black_rainbow);  
%clim([-5.3,5.2]) % Metamaterial
% clim([-170 170]) % LIC

% clim([-40 40]) % nano particle 181404 primary
% clim([-110 110]) % nano particle 181404 secondary
% clim([-150 150]) % nano particle 171925 primary
% clim([-110 110]) % nano particle 171925 secondary


%  colormap(turbo); caxis([-20 20]);
% ax1.XTick = [100,200,300]; % Unit: px
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; % ax1.TickLabelInterpreter = 'latex'; 

%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*10)/10, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*10)/10, length(ax1.YTick) )' );

cb2 = colorbar('Position',[.17+0.685+0.012 .11+.128 .03 .557 ]); % cb2.TickLabelInterpreter = 'latex';
 

% cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 2) dispy v ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes;  
try h1=imshow( flipud(Img_temp) ,'InitialMagnification','fit');
catch h1=surf(  flipud(Img_temp),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,disp_v,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency);  colormap(turbo); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(cMap); caxis([-40 ,40]); % colormap(turbo);  
% caxis([-0.7,0.7]);
% caxis([-0.7,0.7]); % caxis([-0.025,0.025]); 
% colormap(black_rainbow);    caxis([-0.5,0]);
%   colormap(turbo); caxis([-20 20]);
% clim([ -8.5 , 0 ])% Indentation
%clim([ -15 , 15 ]) % Bio-printing
%clim([-9.4,0.4]) % Metamaterial


% clim([-30 20]) % nano particle 181404 primary
% clim([-110 110]) % nano particle 181404 secondary
% clim([-110 50]) % nano particle 171925 primary
% clim([-110 110]) % nano particle 171925 secondary

%clim([-160 160]) % LIC

% ax1.XTick = [100,200,300]; % Unit: px
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; % ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*10)/10, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*10)/10, length(ax1.YTick) )' );
% cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';
cb2 = colorbar('Position',[.17+0.685+0.012 .11+.128 .03 .557 ]); % cb2.TickLabelInterpreter = 'latex';




