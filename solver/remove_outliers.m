function [u,v,cc,BadptRow,BadptCol,RemoveOutliersList] = remove_outliers(u,v,cc,qDICOrNot,Thr0,varargin)
% =========================================================================
% Objective: to manually remove outliers or remove outliers using the
% universal outlier test based on
%
% J. Westerweel and F. Scarano. Universal outlier detection for PIV data.
% Exp. Fluids, 39(6):1096{1100, August 2005. doi: 10.1007/s00348-005-0016-6
% -------------------------------------------------------------------------
% CALL SITES:
%   solver/integer_search.m (line 122): qDICOrNot=0, Thr0=100
%   In this mode, all interactive sections (input/ginput) are bypassed.
% -------------------------------------------------------------------------
% NOTES
% -------------------------------------------------------------------------
% needs medFilt3 and John D'Errico's inpaint_nans3
% (http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint-nans)function.
% =========================================================================

%% Find u,v indices where are holes
HolePtInd = find(isnan(u)); 
 
%%
switch nargin
    case 7
        BadptRow = varargin{1}; BadptCol = varargin{2};
    otherwise
        BadptRow = []; BadptCol = [];
end

% ============== qDIC bad points removal ===============
% prompt = 'Input threshold for qDIC bad points removal:';
% ccThreshold = input(prompt); % Thr = 50;
if qDICOrNot == 1
    
    [M, N] = size(u); mSize = [M*N ,1]; % size of the displacement field
    [cc, ccMask_] = removeBadCorrelations(cc,cc.ccThreshold,1,mSize);
    for ii = 1:2
        ccMask{ii} = reshape(double(ccMask_(:,ii)),mSize);
    end
    qDICpceRmList = ismissing(ccMask{1});
    qDICppeRmList = ismissing(ccMask{2});
else 
    qDICpceRmList = []; 
    qDICppeRmList = [];
end
 

% ============== median test bad points removal ===============
medianU = cell(1,2);
normFluct = cell(1,2);
normFluctMag = zeros(size(u));
epsilon = 0.1;
[medianU{1}, normFluct{1}] = funMedRemoveOutliers(u,epsilon);
[medianU{2}, normFluct{2}] = funMedRemoveOutliers(v,epsilon);
normFluctMag =  normFluct{1}.^2 + normFluct{2}.^2  ;
normFluctMag = sqrt(normFluctMag);

MedFilterOrNot = 0;
showFigures = isempty(Thr0) || (Thr0 == 0);
while MedFilterOrNot < 1

    if showFigures, figure, surf(normFluctMag); axis equal; axis tight; view(2); caxis auto; colorbar; end
	if showFigures
		prompt = '--- Threshold for median test --- Input here: ';
		Thr = input(prompt); % Thr = 50;
	else
		Thr = Thr0;
	end
    RemoveOutliersList = find( normFluctMag > Thr ); % detection criterion

    % ============== remove bad points ===============
    u2 = u; u2(qDICpceRmList) = NaN; u2(qDICppeRmList) = NaN; u2(RemoveOutliersList) = NaN;
    v2 = v; v2(qDICpceRmList) = NaN; v2(qDICppeRmList) = NaN; v2(RemoveOutliersList) = NaN;
    u2 = inpaint_nans(u2,4); v2 = inpaint_nans(v2,4);
    % --------------------------------------
    u2(HolePtInd) = nan;  v2(HolePtInd) = nan; % Set points where is a hole as nans
    if showFigures
        close all;
        figure; surf(u2); colorbar; title('Displacement u','fontweight','normal');
        colormap(jet); set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex';
        figure; surf(v2); colorbar; title('Displacement v','fontweight','normal'); colormap(jet);
        colormap(jet); set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex';
    end

    if showFigures
        fprintf('Do you want to redo Median test: 0(Yes, redo it!); 1(No, it is good!)  \n')
        prompt = 'Input here: ';
        MedFilterOrNot = input(prompt); % Thr = 50;
    else
        MedFilterOrNot = 1;
    end

end
u = u2; v = v2;
 

%%%%% Phi test: to remove bad points where cross correlation value is small %%%%%
PhiFilterOrNot = 1; 
while PhiFilterOrNot < 1
    
    figure, surf(cc.max); axis equal; axis tight; view(2); caxis auto; colorbar;
	if  (PhiFilterOrNot == 0)
		prompt = '--- Threshold for Phi test --- Input here: ';
		PhiThr = input(prompt); % Thr = 50;
	else
		PhiThr = PhiThr0;
    end
    try 
        RemoveOutliersList = find( cc.max < PhiThr ); % detection criterion
    catch
        RemoveOutliersList = [];
    end
    % ============== remove bad points ===============
    u2 = u;  u2(RemoveOutliersList) = NaN;
    v2 = v;  v2(RemoveOutliersList) = NaN;
    u2 = inpaint_nans(u2,4); v2 = inpaint_nans(v2,4);
    % --------------------------------------
    close all;
    u2(HolePtInd) = nan;  v2(HolePtInd) = nan; % Set points where is a hole as nans
    figure; surf(u2); colorbar; title('Displacement u','fontweight','normal'); 
    colormap(jet); set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    figure; surf(v2); colorbar; title('Displacement v','fontweight','normal'); colormap(jet);
    colormap(jet); set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    
    try 
        if PhiFilterOrNot == 0  
            fprintf('Do you want to redo Phi test: 0(Yes, redo it!); 1(No, it is good!)  \n')
            prompt = 'Input here: ';
            PhiFilterOrNot = input(prompt);  
        else
            PhiFilterOrNot = 1;
        end
    catch
        PhiFilterOrNot = 1;
    end
    
    
end

u = u2; v = v2;

 


%% ==============================================
% Manually remove bad points.
if abs(qDICOrNot) > 0
fprintf('Do you clear bad points by setting upper/lower bounds? (0-yes; 1-no)  \n')
prompt = 'Input here: ';
ClearBadInitialPointsOrNot = input(prompt);

while ClearBadInitialPointsOrNot == 0
    
    prompt = 'What is the upper bound for x-disp? Input: ';
    upperbound = input(prompt);
    [row1,col1] = find(u>upperbound);
    prompt = 'What is the lower bound for x-disp? Input: ';
    lowerbound = input(prompt);
    [row2,col2] = find(u<lowerbound);
    prompt = 'What is the upper bound for y-disp? Input: ';
    upperbound = input(prompt);
    [row3,col3] = find(v>upperbound);
    prompt = 'What is the lower bound for y-disp? Input: ';
    lowerbound = input(prompt);
    [row4,col4] = find(v<lowerbound);
    
    row = [row1; row2; row3; row4]; col = [col1; col2; col3; col4];
    
    for tempi = 1:length(row)
        u(row(tempi),col(tempi))=NaN; v(row(tempi),col(tempi))=NaN;
        %f11(row(tempi),col(tempi))=NaN; f21(row(tempi),col(tempi))=NaN;
        %f12(row(tempi),col(tempi))=NaN; f22(row(tempi),col(tempi))=NaN;
    end
    
    u = inpaint_nans(u,4); v = inpaint_nans(v,4);
    %f11 = inpaint_nans(f11,4); f21 = inpaint_nans(f21,4);
    %f12 = inpaint_nans(f12,4); f22 = inpaint_nans(f22,4);
    
    % --------------------------------------
    close all;
    u(HolePtInd) = nan;  v(HolePtInd) = nan; % Set points where is a hole as nans
    figure; surf(u); colorbar; title('Displacement u','fontweight','normal');
    colormap(jet);  set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    figure; surf(v); colorbar; title('Displacement v','fontweight','normal');
    colormap(jet);  set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    
    prompt = 'Do you want to reset upper/lower bounds? (0-yes; 1-no) Input: ';
    ClearBadInitialPointsOrNot = input(prompt);
end


%% =========
fprintf('Do you clear bad points by directly clicking x-disp bad points? (0-yes; 1-no)  \n')
prompt = 'Input here: ';
ClearBadInitialPointsOrNot = input(prompt);

while ClearBadInitialPointsOrNot == 0
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Have a look at integer search
    % --------------------------------------
    close all;
    figure; surf(u); colorbar; view(2)
    title('Displacement u','fontweight','normal')
    colormap(jet);  set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [row1, col1] = ginput; row = floor(col1); col = floor(row1); row=row(:); col=col(:);
    BadptRow=[BadptRow;row]; BadptCol=[BadptCol;col]; row=BadptRow; col=BadptCol;
    for tempi = 1:length(row)
        if row(tempi)>0 && col(tempi)>0 && row(tempi)<size(u,1) && col(tempi)<size(u,2)
        u(row(tempi),col(tempi))=NaN; v(row(tempi),col(tempi))=NaN;
        %f11(row(tempi),col(tempi))=NaN; f21(row(tempi),col(tempi))=NaN;
        %f12(row(tempi),col(tempi))=NaN; f22(row(tempi),col(tempi))=NaN;
        end
    end
    u = inpaint_nans(u,4); v = inpaint_nans(v,4);
    %f11 = inpaint_nans(f11,4); f21 = inpaint_nans(f21,4);
    %f12 = inpaint_nans(f12,4); f22 = inpaint_nans(f22,4);
    
    % --------------------------------------
    close all;
    u(HolePtInd) = nan; v(HolePtInd) = nan; % Set points where is a hole as nans
    figure; surf(u); colorbar; title('Displacement u','fontweight','normal');
    colormap(jet);  set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    
    prompt = 'Do you want to remove more x-disp bad points? (0-yes; 1-no) Input: ';
    ClearBadInitialPointsOrNot = input(prompt);
    
end

%prompt = 'Do you clear bad points by directly pointing y-disp bad points? (0-yes; 1-no)';
fprintf('Do you clear bad points by directly clicking y-disp bad points? (0-yes; 1-no)  \n')
prompt = 'Input here: ';
ClearBadInitialPointsOrNot = input(prompt);

while ClearBadInitialPointsOrNot == 0
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Have a look at integer search
    % --------------------------------------
    close all;
    figure; surf(v); colorbar; view(2)
    title('Displacement v','fontweight','normal')
    colormap(jet); set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [row1, col1] = ginput; row = floor(col1); col = floor(row1); row=row(:); col=col(:);
    BadptRow=[BadptRow;row]; BadptCol=[BadptCol;col]; row=BadptRow; col=BadptCol;
    
    for tempi = 1:length(row)
        if row(tempi)>0 && col(tempi)>0 && row(tempi)<size(u,1) && col(tempi)<size(u,2)
        u(row(tempi),col(tempi))=NaN; v(row(tempi),col(tempi))=NaN;
        %f11(row(tempi),col(tempi))=NaN; f21(row(tempi),col(tempi))=NaN;
        %f12(row(tempi),col(tempi))=NaN; f22(row(tempi),col(tempi))=NaN;
        end
    end
    u = inpaint_nans(u,4); v = inpaint_nans(v,4);
    %f11 = inpaint_nans(f11,4); f21 = inpaint_nans(f21,4);
    %f12 = inpaint_nans(f12,4); f22 = inpaint_nans(f22,4);
    
    % --------------------------------------
    close all;
    u(HolePtInd) = nan; v(HolePtInd) = nan; % Set points where is a hole as nans
    figure; surf(v); colorbar; title('Displacement v','fontweight','normal');
    colormap(jet);  set(gca,'fontsize',18); set(gcf,'color','w'); a = gca; a.TickLabelInterpreter = 'latex'; 
    
    % prompt = 'Do you clear bad points by directly pointing y-disp bad points more? (0-yes; 1-no)';
    prompt = 'Do you want to remove more y-disp bad points? (0-yes; 1-no) Input: ';
    ClearBadInitialPointsOrNot = input(prompt);
    
end

end
end

%% ========================================================================
function [medianU, normFluct] = funMedRemoveOutliers(u,epsilon)

inpaint_opt = 3;

nSize = 2*[1 1];
skipIdx = ceil(prod(nSize)/2);
padOption = 'symmetric';

u =  inpaint_nans(double(u),inpaint_opt);

medianU = medFilt2(u,nSize,padOption,skipIdx);
fluct = u - medianU;
medianRes = medFilt2(abs(fluct),nSize,padOption,skipIdx);
normFluct = abs(fluct./(medianRes + epsilon));


end

%% ========================================================================
function Vr = medFilt2(V0,nSize, padoption, skipIdx)
% fast median filter for 2D data with extra options.

if nargin < 4, skipIdx = 0; end
if nargin < 3, padoption = 'symmetric'; end
if nargin < 2, nSize = [2 2]; end

nLength = prod(nSize);
if mod(nLength,2) == 1, padSize = floor(nSize/2);
elseif mod(nLength,2) == 0, padSize = [nSize(1)/2-1,nSize(2)/2];
end

if strcmpi(padoption,'none')
    V = V0;
else
    V = (padarray(V0,padSize(1)*[1,1],padoption,'pre'));
    V = (padarray(V,padSize(2)*[1,1],padoption,'post'));
end

S = size(V);
nLength = prod(nSize)-sum(skipIdx>1);
Vn = single(zeros(S(1)-(nSize(1)-1),S(2)-(nSize(2)-1),nLength));  % all the neighbor

%%
% build the neighborhood

i = cell(1,nSize(1)); j = cell(1,nSize(2));
for m = 1:nSize(1), i{m} = m:(S(1)-(nSize(1)-m)); end
for m = 1:nSize(2), j{m} = m:(S(2)-(nSize(2)-m)); end

p = 1;
for m = 1:nSize(1)
    for n = 1:nSize(2)
        if p ~= skipIdx || skipIdx == 0
            Vn(:,:,p) = V(i{m},j{n});
        end
        p = p + 1;
    end
end

if skipIdx ~= 0, Vn(:,:,skipIdx) = []; end
% perform the processing
Vn = sort(Vn,3);

if mod(nLength,2) == 1 % if odd get the middle element
    Vr = Vn(:,:,ceil(nLength/2));
else % if even get the mean of the two middle elements
    Vr = mean(cat(3,Vn(:,:,nLength/2),Vn(:,:,nLength/2+1)),4);
end

end

%% ========================================================================
function [cc, ccMask] = ...
    removeBadCorrelations(cc,ccThreshold,sizeChange,mSize)

if sizeChange == 1
    %recompute threshold, only use pce & ppe since these give the best
    %results emprically.
    for ii = 1:2
        
        [qf_para{ii},single_distro] = bimodal_gauss_fit(cc.qfactors(:,ii));
        
        if single_distro == 0%(qf_para{ii}(2) + 2*qf_para{ii}(4)) < (qf_para{ii}(3) - 2*qf_para{ii}(5))
            cc.q_thresh{ii} = qf_para{ii}(3) - ccThreshold*qf_para{ii}(5);
        elseif single_distro == 1
            cc.q_thresh{ii} = qf_para{ii}(3) - ccThreshold*qf_para{ii}(5);
        else
            cc.q_thresh{ii} = qf_para{ii}(3) - ccThreshold*qf_para{ii}(5);
        end
    end
    q_trim = [cc.q_thresh{1};cc.q_thresh{2}];
else
    q_trim = [cc.q_thresh{1};cc.q_thresh{2}];
end

%NaN the qfactor values that are below the threshold
temp = bsxfun(@le,cc.qfactors(:,1:2),q_trim');
qfactors_accept = cc.qfactors(:,1:2);
qfactors_accept(temp) = NaN;

for ii = 1:2
    cc.qfactors_accept{ii} = reshape(double(qfactors_accept(:,ii)),mSize);
end

ccMask = ones(size(qfactors_accept)) + ...
    zeros(size(qfactors_accept)).*qfactors_accept;

end



%% ========================================================================
function [paramEsts,single_distro] = bimodal_gauss_fit(x)
%This function takes a dataset and fits a bimodal Gaussian distro to it.

x = sort(x);

%set function for bimodal Gaussian
pdf_normmixture = @(x,p,mu1,mu2,sigma1,sigma2) ...
    p*normpdf(x,mu1,sigma1) + (1-p)*normpdf(x,mu2,sigma2);
pdf_single = @(x,mu1,sigma1) ...
    normpdf(x,mu1,sigma1);

%starting params, biased mixture toward "good" values,
%centered at quartiles, equal std dev.
pStart = 0.25;
muStart = quantile(x,[.10 .75]);
sigmaStart(1) = sqrt(var(x(1:round(length(x)/5))));
%- 0.25*diff(quantile(x,[0.01 0.25])).^2);
sigmaStart(2) = sqrt(var(x(ceil(length(x)/10):ceil(3*length(x)/4))));
%... - 0.25*diff(quantile(x,[0.25 0.75])).^2);%1:round(length(x)/2)
start = [pStart muStart sigmaStart];

%set lower and upper bounds
lb = [0 -inf -inf 0.00001 0.00001];
ub = [1 inf inf inf inf];

%do the parameter estimation
options = statset('MaxIter',1800, 'MaxFunEvals',3600);
% options.FunValCheck = 'off';
try
    single_distro = 0;
    paramEsts = mle(x, 'pdf',pdf_normmixture, 'start',start, ...
        'lower',lb, 'upper',ub, 'options',options);%,'optimfun','fmincon'
    
    if paramEsts(2)-paramEsts(4) >= paramEsts(3)+paramEsts(5) || ...
            paramEsts(2)+paramEsts(4) <= paramEsts(3)-paramEsts(5)
        
        single_distro = 1;
        %     disp('Parameters estimated for single peak Gaussian')
        paramEsts = mle(x,'options',options);%,'optimfun','fmincon'
        paramEsts = [0.5,paramEsts(1),paramEsts(1),paramEsts(2),...
            paramEsts(2)];
        
    end
    
catch
    single_distro = 1;
    %     disp('Parameters estimated for single peak Gaussian')
    paramEsts = mle(x,'options',options);%,'optimfun','fmincon'
    paramEsts = [0.5,paramEsts(1),paramEsts(1),paramEsts(2),...
        paramEsts(2)];
end

% % %show the result
% % figure
% % % [~, bins] =
% % histogram(x,100);
% % % bins = -2.5:.5:7.5;
% % % h = bar(bins,histc(x,bins)/(length(x)*0.5),'histc');
% % % histogram(x,100)
% % % h.FaceColor = [0.9 0.9 0.9];
% % xgrid = linspace(1.1*min(x),1.1*max(x),200);
% % pdfgrid = pdf_normmixture(xgrid,paramEsts(1),paramEsts(2),paramEsts(3),...
% %     paramEsts(4),paramEsts(5));
% % hold on
% % plot((paramEsts(3) - 2*paramEsts(5)),pdfgrid,'or')
% % plot((paramEsts(2) + 2*paramEsts(4)),pdfgrid,'*r')
% % plot(xgrid,pdfgrid,'-b')
% % hold off
% % xlabel('x')
% % ylabel('Probability Density')

end




