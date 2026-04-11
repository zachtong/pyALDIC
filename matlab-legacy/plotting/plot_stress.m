function [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
    stress_principal_min_xyplane, stress_maxshear_xyplane, ...
    stress_maxshear_xyz3d, stress_vonMises] = plot_stress(DICpara,ResultStrain,coordinatesFEM,elementsFEM,CurrentImg)
%plot_stress: to compute and plot DIC solved stress fields on the original DIC images
%  [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
%   stress_principal_min_xyplane, stress_maxshear_xyplane, ...
%   stress_maxshear_xyz3d, stress_vonMises] = PlotstressQuadtree(DICpara,ResultStrain,coordinatesFEM,elementsFEM,CurrentImg)
%
%   INPUT: DICpara          DIC para in the ALDIC code
%          ResultStrain     DIC computed strain field result
%          CurrentImg       File name of current deformed image
%
%   OUTPUT: stress_sxx      Cauchy stress xx-compoent
%           stress_sxy      Cauchy stress xy-compoent s_xy = s_yx
%           stress_syy      Cauchy stress yy-compoent
%           stress_principal_max_xyplane    max principal stress on the xy-plane
%           stress_principal_min_xyplane    min principal stress on the xy-plane
%           stress_maxshear_xyplane         max shear stress on the xy-plane
%           stress_maxshear_xyz3d           max shear stress on the xyz-three dimensional space
%           stress_vonMises                 von Mises stress
%
%   Plots:       
%       1) stress sxx
%       2) stress sxy
%       3) stress syy
%       4) max principal stress on the xy-plane 
%       5) min principal stress on the xy-plane
%       6) max shear stress on the xy-plane
%       7) max shear stress on the xyz-three dimensional space
%       8) equivalent von Mises stress
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
% Last time updated: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
warning off; load('./plotting/colormap_RdYlBu.mat','cMap');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% convert pixel unit to the physical world unit %%%%%
um2px = DICpara.um2px; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OrigDICImgTransparency = DICpara.OrigDICImgTransparency; % Original raw DIC image transparency
Image2PlotResults = DICpara.Image2PlotResults; % Choose image to plot over (first only, second and next images)
   
%% Load computed strain fields
dudx = ResultStrain.dudx; dvdx = ResultStrain.dvdx; 
dudy = ResultStrain.dudy; dvdy = ResultStrain.dvdy; 
strain_exx = dudx; 
strain_exy = 0.5*(dvdx + dudy); 
strain_eyy = dvdy;

%% Load displacement components to deform the reference image
disp_u = ResultStrain.dispu; 
disp_v = ResultStrain.dispv; 
coordinatesFEMWorldDef = coordinatesFEM;

%% Compute stress components

% ------ For each material model ------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DICpara.MaterialModel == 1 
    
    % Linear elasticity -- Plane stress
    % stress_xyz = [sxx  sxy  0;      strain_xyz = [exx  exy  0;
    %               sxy  syy  0;                    exy  eyy  0;
    %               0    0    0];                   0    0    ezz];
    E = DICpara.MaterialModelPara.YoungsModulus;
    nu = DICpara.MaterialModelPara.PoissonsRatio;
    stress_sxx = E/(1-nu^2)*(strain_exx + nu*strain_eyy);
    stress_syy = E/(1-nu^2)*(nu*strain_exx + strain_eyy);
    stress_sxy = E/(1+nu)*strain_exy;
     
    % Principal stress
    stress_maxshear_xyplane = sqrt((0.5*(stress_sxx-stress_syy)).^2 + stress_sxy.^2);
    stress_principal_max_xyplane = 0.5*(stress_sxx+stress_syy) + stress_maxshear_xyplane;
    stress_principal_min_xyplane = 0.5*(stress_sxx+stress_syy) - stress_maxshear_xyplane;
    stress_maxshear_xyz3d = max(stress_maxshear_xyplane, 0.5*abs(stress_principal_max_xyplane));
    
    % von Mises stress
    stress_vonMises = sqrt(0.5*( (stress_principal_max_xyplane-stress_principal_min_xyplane).^2 + ...
        (stress_principal_max_xyplane).^2 + (stress_principal_min_xyplane).^2 ));
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
elseif DICpara.MaterialModel == 2 
    
    % Linear elasticity -- Plane strain
    % stress_xyz = [sxx  sxy  0;      strain_xyz = [exx  exy  0;
    %               sxy  syy  0;                    exy  eyy  0;
    %               0    0    szz];                 0    0    0]; 
    E = DICpara.MaterialModelPara.YoungsModulus;
    nu = DICpara.MaterialModelPara.PoissonsRatio;
    stress_sxx = E*(1-nu)/(1+nu)/(1-2*nu)*(strain_exx + nu/(1-nu)*strain_eyy);
    stress_syy = E*(1-nu)/(1+nu)/(1-2*nu)*(strain_eyy + nu/(1-nu)*strain_exx);
    stress_sxy = E/(1+nu)*strain_exy;
    stress_szz = nu*(stress_sxx + stress_syy);

    % Principal stress
    stress_maxshear_xyplane = sqrt((0.5*(stress_sxx-stress_syy)).^2 + stress_sxy.^2);
    stress_principal_max_xyplane = 0.5*(stress_sxx+stress_syy) + stress_maxshear_xyplane;
    stress_principal_min_xyplane = 0.5*(stress_sxx+stress_syy) - stress_maxshear_xyplane;
    stress_maxshear_xyz3d = reshape(  max( [ stress_maxshear_xyplane(:), 0.5*abs(stress_principal_max_xyplane(:)-stress_szz(:)), ...
                                0.5*abs(stress_principal_min_xyplane(:)-stress_szz(:)) ], [], 2 ),  size(stress_maxshear_xyplane) ) ;
                            
    % von Mises stress
    stress_vonMises = sqrt(0.5*( (stress_principal_max_xyplane-stress_principal_min_xyplane).^2 + ...
        (stress_principal_max_xyplane-stress_szz).^2 + (stress_principal_min_xyplane-stress_szz).^2 ));
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
elseif DICpara.MaterialModel == 3 % Neo-Hookean or other material models
    
    disp('User needs to modify the code by yourself.'); pause; 
    % TODO: Please modify by yourself
    
end

 

        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 1) Stress sxx ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg) ),'InitialMagnification','fit');
catch h1=surf( flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_sxx,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([-4e8,4e8]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';


 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 2) Strain sxy ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf( flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_sxy,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([-5e8,5e8]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';

 

 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 3) Strain syy ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf( flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_syy,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([-1e8,1.3e9]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';


 

 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 4) Strain stress_principal_max_xyplane ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf(  flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_principal_max_xyplane,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([-1e8,1.3e9]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';




 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 5) Strain stress_principal_min_xyplane ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf(  flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_principal_min_xyplane,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([-4e8,4e8]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';




 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 6) Strain stress_maxshear_xyplane ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf(  flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_maxshear_xyplane,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([0,0.6e9]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';



 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 7) Strain stress_maxshear_xyz3d ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf(  flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2); set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_maxshear_xyz3d,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([0,0.7e9]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';


 
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 8) von Mises stress ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig1=figure; ax1=axes; 
try h1=imshow( flipud(imread(CurrentImg)),'InitialMagnification','fit');
catch h1=surf(  flipud( imread(CurrentImg) ),'EdgeColor','none','LineStyle','none');
end

axis on; axis equal; axis tight; box on; set(gca,'fontSize',18); view(2);  set(gca,'ydir','normal');
hold on; ax2=axes; h2=show([],elementsFEM(:,1:4),coordinatesFEMWorldDef/um2px,stress_vonMises,'NoEdgeColor');
set(gca,'fontSize',18); view(2); box on; axis equal;  axis tight;   
alpha(h2,OrigDICImgTransparency); colormap(jet); caxis auto;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet); 
% caxis([0,1.2e9]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linkaxes([ax1,ax2]);  % Link axes together
ax2.Visible = 'off'; ax2.XTick = []; ax2.YTick = []; % Hide the top axes
colormap(ax1,'gray'); % Give each one its own colormap
set([ax1,ax2],'Position',[.17 .11 .685 .815]);  
ax1.Visible = 'on'; ax1.TickLabelInterpreter = 'latex'; 
%%%%% convert pixel unit to the physical world unit %%%%%
xticklabels(ax1, num2cell(round(um2px*ax1.XTick*100)/100, length(ax1.XTick) )' );
yticklabels(ax1, num2cell(round(um2px*ax1.YTick*100)/100, length(ax1.YTick) )' );
cb2 = colorbar('Position',[.17+0.685+0.012 .11 .03 .815]); cb2.TickLabelInterpreter = 'latex';






end
 
 