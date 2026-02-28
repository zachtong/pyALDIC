function plot_strain_show(F,coordinatesFEM,elementsFEM,varargin)
%FUNCTION plot_strain_show(F,coordinatesFEM,elementsFEM)
% To plot DIC solved strain components
% ----------------------------------------------
%
%   INPUT: F                 Deformation gradient tensor: 
%                            F = [F11_node1, F21_node1, F12_node1, F22_node1, ... , F11_nodeN, F21_nodeN, F12_nodeN, F22_nodeN]';
%          coordinatesFEM    FE mesh coordinates
%          elementsFEM       FE mesh elements
%          DICpara           chosen DIC parameters
%          EdgeColorOrNot    show edge color or not

%   OUTPUT: Plots of exx, eyy, and exy strain fields.
%
% ----------------------------------------------
% Author: Jin Yang.  
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Last date modified: 2020.12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%% Initialization
warning off;  F = full(F);

%%%%% Parse Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[DICpara,EdgeColorOrNot] = parseargs(varargin);

%%%%% convert pixel unit to the physical world unit %%%%%
try um2px = DICpara.um2px; 
catch um2px = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try EdgeColorOrNot = EdgeColorOrNot;
catch EdgeColorOrNot = 'EdgeColor';
end


 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 1) strain exx ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; show([],elementsFEM(:,1:4),coordinatesFEM,F(1:4:end),EdgeColorOrNot); 
title('Strain $e_{11}$','fontweight','normal','Interpreter','latex'); set(gca,'fontsize',18); 
view(2); axis tight; axis equal;  box on;  set(gcf,'color','w'); 
colorbar; colormap jet;

if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';



 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 2) strain exy ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; show([],elementsFEM(:,1:4),coordinatesFEM,0.5*(F(2:4:end)+F(3:4:end)),EdgeColorOrNot); 
title('Strain $e_{12}$','fontweight','normal','Interpreter','latex'); set(gca,'fontsize',18); 
view(2); axis tight; axis equal;  box on;  set(gcf,'color','w'); 
colorbar; colormap jet;

if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';


 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ====== 3) strain eyy ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; show([],elementsFEM(:,1:4),coordinatesFEM,F(4:4:end),EdgeColorOrNot); 
title('Strain $e_{22}$','fontweight','normal','Interpreter','latex'); set(gca,'fontsize',18); 
view(2); axis tight; axis equal;  box on;  set(gcf,'color','w'); 
colorbar; colormap jet;

if um2px==1, xlabel('$x$ (pixels)','Interpreter','latex'); ylabel('$y$ (pixels)','Interpreter','latex');
else, xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
end

a = gca; a.TickLabelInterpreter = 'latex';
b = colorbar; b.TickLabelInterpreter = 'latex';


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