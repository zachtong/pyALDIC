function plot_disp_show(U,coordinatesFEM,elementsFEM,varargin)
%plot_disp_show: to plot DIC solved displacement components
%   plot_disp_show(U,coordinatesFEM,elementsFEM)
% ----------------------------------------------
%
%   INPUT: U                 Displacement vector: 
%                            U = [Ux_node1, Uy_node1, Ux_node2, Uy_node2, ... , Ux_nodeN, Uy_nodeN]';
%          coordinatesFEM    FE mesh coordinates
%          elementsFEM       FE mesh elements
%          DICpara           chosen DIC parameters
%          EdgeColorOrNot    show edge color or not
%
%   OUTPUT: Plots of x-displacement field and y-displacement field.
%
%   TODO: users could change caxis range based on their own choices.
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last date modified: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initialization
warning off;  U = full(U);
% run('./img_Vito/Black_rainbow.m');

%%%%% Parse Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[DICpara,EdgeColorOrNot] = parseargs(varargin);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% convert pixel unit to the physical world unit %%%%%
try um2px = DICpara.um2px; 
catch um2px = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try EdgeColorOrNot = EdgeColorOrNot;
catch EdgeColorOrNot = 'EdgeColor';
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 1) dispx u ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; show([],elementsFEM(:,1:4),coordinatesFEM,U(1:2:end),EdgeColorOrNot); 

title('$x-$displacement $u$','FontWeight','Normal','Interpreter','latex');
view(2); set(gca,'fontsize',18); axis tight; axis equal; colorbar; % view([90 -90])
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end
set(gcf,'color','w'); colormap jet; box on;
a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';

%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
% caxis([-0.56,0.56]); % colorturbo = turbo(64); colorturbo(:,2)=0; colormap(turbo)
%  colormap(black_rainbow_plus ); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 2) dispx v ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; show([],elementsFEM(:,1:4),coordinatesFEM,U(2:2:end),EdgeColorOrNot); 

title('$y-$displacement $v$','FontWeight','Normal','Interpreter','latex');
view(2); set(gca,'fontsize',18); axis tight; axis equal; colorbar;
if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end
set(gcf,'color','w'); colormap jet; box on;
a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';

%%%%%% TODO: manually modify colormap and caxis %%%%%%
% colormap(jet);  % D Sample 
% caxis([-0.1,0.1]) % foam
% caxis([-0.004,0.004]); % Sample 12
% caxis([0,0.1]); 
% caxis([-0.12,0.12]); colormap(black_rainbow_plus); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [DICpara,EdgeColorOrNot] = parseargs(vargin)
DICpara=[]; EdgeColorOrNot=[];  
 
try 
    DICpara=vargin{1};
    try
        EdgeColorOrNot=vargin{2};
    catch
    end
catch
    
end

end
