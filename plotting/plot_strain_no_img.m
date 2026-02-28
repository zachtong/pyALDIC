function [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
    strain_maxshear,strain_vonMises] = plot_strain_no_img(F,coordinatesFEMWorld,elementsFEM,DICpara)
%plot_strain_no_img: to plot DIC solved strain fields in a quadtree mesh
%   [strain_exx,strain_exy,strain_eyy,strain_principal_max,strain_principal_min, ...
%    strain_maxshear,strain_vonMises] = plot_strain_no_img(F,coordinatesFEM,elementsFEM)
%
%   INPUT: F                    DIC solved deformation gradient tensor
%          coordinatesFEMWorld  coordinates of each points on the image domain
%          elementsFEM          FE-elements
%          DICpara              chosen DIC parameters
%
%   OUTPUT: strain_exx              strain xx-compoent
%           strain_exy              strain xy-compoent
%           strain_eyy              strain yy-compoent
%           strain_principal_max    max principal strain on the xy-plane
%           strain_principal_min    min principal strain on the xy-plane
%           strain_maxshear         max shear strain on the xy-plane
%           strain_vonMises         equivalent von Mises strain
%
%   Plots:       
%       1) strain sxx
%       2) strain sxy
%       3) strain syy
%       4) max principal strain on the xy-plane 
%       5) min principal strain on the xy-plane
%       6) max shear strain on the xy-plane
%       7) equivalent von Mises strain
%
%
% ----------------------------------------------
% Author: Jin Yang.
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last time updated: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
warning off; load('./plotting/colormap_RdYlBu.mat','cMap');
% run('./img_Vito/Black_rainbow.m');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% convert pixel unit to the physical world unit %%%%%
um2px = DICpara.um2px; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute strain components

u_x = F(1:4:end); v_x = F(2:4:end);
u_y = F(3:4:end); v_y = F(4:4:end);

strain_exx = u_x; 
strain_exy = 0.5*(v_x+u_y);
strain_eyy = v_y;

strain_maxshear = sqrt((0.5*(strain_exx-strain_eyy)).^2 + strain_exy.^2);
% Principal strain
strain_principal_max = 0.5*(strain_exx+strain_eyy) + strain_maxshear;
strain_principal_min = 0.5*(strain_exx+strain_eyy) - strain_maxshear;
% equivalent von Mises strain
strain_vonMises = sqrt(strain_principal_max.^2 + strain_principal_min.^2 - ...
             strain_principal_max.*strain_principal_min + 3*strain_maxshear.^2);
         
         
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 1) Strain exx ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_exx,'NoEdgeColor');
% showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_exx);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Strain $e_{xx}$','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';
 
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
% colormap(black_rainbow_plus); caxis([-0.004,0.004]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 2) Strain exy ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_exy,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_exy);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Strain $e_{xy}$','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';
 
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
% colormap(black_rainbow_plus); caxis([-0.0018,0.0018]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 3) Strain eyy ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_eyy,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_eyy);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Strain $e_{yy}$','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';
  
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
% colormap(black_rainbow_plus); caxis([-0.0021,0.0021]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 4) Strain e_principal_max ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_principal_max,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_principal_max);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Principal strain $e_{\max}$','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';

%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 5) Strain e_principal_min ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_principal_min,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_principal_min);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Principal strain $e_{\min}$','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';

%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 6) Strain e_max_shear ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_maxshear,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_maxshear);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('Max shear strain','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';
 
%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% ====== 7) von Mises equivalent strain ====== 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;  
show([],elementsFEM(:,1:4),coordinatesFEMWorld,strain_vonMises,'NoEdgeColor');
%showQuadtree(elementsFEM(:,1:4),coordinatesFEMWorld,strain_vonMises);
set(gca,'fontSize',18); view(2); box on; set(gca,'ydir','normal');
title('von Mises equivalent strain','FontWeight','Normal','Interpreter','latex');
axis tight; axis equal; colorbar; colormap jet; set(gcf,'color','w');
if max(coordinatesFEMWorld(:,1)) < um2px*200, set(gca,'XTick',[]); end
if max(coordinatesFEMWorld(:,2)) < um2px*200, set(gca,'YTick',[]); end
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';

%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



end
