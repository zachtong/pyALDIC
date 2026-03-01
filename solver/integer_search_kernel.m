%% Integer Search

function [x,y,u,v,cc] = integer_search_kernel(f,g,tempSizeOfSearchRegion,gridx,gridy,winsize,winstepsize,tempNoOfInitPt,varargin)

if length(winsize)==1, winsize = winsize*[1,1]; end
if length(winstepsize)==1, winstepsize = winstepsize*[1,1]; end

% Parse optional trailing arguments: showWaitbar (logical) and ClusterNo (numeric)
showWaitbar = true;
ClusterNo = 0;
% Extract ClusterNo if last arg is numeric
if ~isempty(varargin) && isnumeric(varargin{end})
    ClusterNo = varargin{end};
    varargin(end) = [];
end
% Extract showWaitbar if next-to-last arg is logical
if ~isempty(varargin) && islogical(varargin{end})
    showWaitbar = varargin{end};
    varargin(end) = [];
end

switch tempNoOfInitPt % 0- whole field; 1- several seeds points;
    case 0 % 0- whole field;
        
        gridxBackup = gridx; gridyBackup = gridy;
        
        % may lose boundary regions
        while gridx(1)-tempSizeOfSearchRegion(1)-0.5*winsize(1) < 1
            gridx(1) = gridx(1) +1;
        end
        while gridy(1)-tempSizeOfSearchRegion(2)-0.5*winsize(2) < 1
            gridy(1) = gridy(1) +1;
        end
        while gridx(end)+1.5*winsize(1)+tempSizeOfSearchRegion(1) > size(f,1)
            gridx(end) = gridx(end)-1;
        end
        while gridy(end)+1.5*winsize(2)+tempSizeOfSearchRegion(2) > size(f,2) 
            gridy(end) = gridy(end)-1;
        end
       
       if gridx(1) >= gridx(end) || gridy(1) >= gridy(end)
            warning('integer_search_kernel:gridDegenerate', ...
                'SizeOfFFTSearchRegion too large for image size. gridx=[%d,%d], gridy=[%d,%d]. Reduce SizeOfFFTSearchRegion.', ...
                gridx(1), gridx(end), gridy(1), gridy(end));
            gridx = gridxBackup; gridy = gridyBackup;
        end
       [x0,y0,u0,v0,cc0] = funIntegerSearchWholeField(f,g,tempSizeOfSearchRegion,gridx,gridy,winsize,winstepsize,showWaitbar,ClusterNo);
        x=x0; y=y0; u=u0; v=v0; cc=cc0;
%         xList = gridxBackup(1)+4:winstepsize:gridxBackup(end)-4; 
%         yList = gridyBackup(1)+4:winstepsize:gridyBackup(end)-4;
%         [x,y] = meshgrid(xList,yList);
%         u = gridfit(x0,y0,u0,xList,yList,'smooth',1,'interp','bilinear','regularizer','springs');
%         v = gridfit(x0,y0,v0,xList,yList,'smooth',1,'interp','bilinear','regularizer','springs');
%         cc.max = gridfit(x0,y0,cc0.max,xList,yList,'smooth',1,'interp','bilinear','regularizer','springs');
        
         
    case 1 % 1- several seeds points;
        seedPtCoords = varargin{1}; uSeedPt = zeros(size(seedPtCoords,1),1); vSeedPt = uSeedPt; PhiSeedPt = uSeedPt;
        for tempi = 1:length(seedPtCoords)
            
            if ( ceil(seedPtCoords(tempi,1)-winsize(1)/2)-tempSizeOfSearchRegion(1) < 1) || ...
                    (floor(seedPtCoords(tempi,1)+winsize(1)/2)+tempSizeOfSearchRegion(1) > size(g,1)) || ...
                    (ceil(seedPtCoords(tempi,2)-winsize(2)/2)-tempSizeOfSearchRegion(2) < 1) || ...
                    (floor(seedPtCoords(tempi,2)+winsize(2)/2)+tempSizeOfSearchRegion(2) > size(g,2))
                continue;
            else
            
                C = f(ceil(seedPtCoords(tempi,1)-winsize(1)/2):floor(seedPtCoords(tempi,1)+winsize(1)/2), ...
                      ceil(seedPtCoords(tempi,2)-winsize(2)/2):floor(seedPtCoords(tempi,2)+winsize(2)/2));
                D = g(ceil(seedPtCoords(tempi,1)-winsize(1)/2)-tempSizeOfSearchRegion(1):floor(seedPtCoords(tempi,1)+winsize(1)/2)+tempSizeOfSearchRegion(1), ...
                      ceil(seedPtCoords(tempi,2)-winsize(2)/2)-tempSizeOfSearchRegion(2):floor(seedPtCoords(tempi,2)+winsize(2)/2)+tempSizeOfSearchRegion(2));  

                XCORRF2OfCD0 = normxcorr2(C,D);

                [v1temp, u1temp, max_f] = findpeak(XCORRF2OfCD0(winsize(1):end-winsize(1)+1, winsize(2):end-winsize(2)+1),1);

                zero_disp = ceil(size(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1))/2);

                uSeedPt(tempi) = u1temp-zero_disp(1);
                vSeedPt(tempi) = v1temp-zero_disp(2);
                PhiSeedPt(tempi) = max_f;
            end
            
        end
        
        xList = gridx(1):winstepsize(1):gridx(end); yList = gridy(1):winstepsize(2):gridy(end);
        [x,y] = meshgrid(xList,yList);
        %bc = Spline2D('bicubic',X,Y,ZZ);
        u = gridfit(seedPtCoords(:,1),seedPtCoords(:,2),uSeedPt,xList,yList,...
            'smooth',100,'interp','bilinear','regularizer','springs');
        v = gridfit(seedPtCoords(:,1),seedPtCoords(:,2),vSeedPt,xList,yList,...
            'smooth',100,'interp','bilinear','regularizer','springs');
        cc.max = gridfit(seedPtCoords(:,1),seedPtCoords(:,2),PhiSeedPt,xList,yList,...
            'smooth',100,'interp','bilinear','regularizer','springs');
        
    otherwise
        disp('wrong input in IntegerSearch!');
end

end

function [x,y,u,v,cc] = funIntegerSearchWholeField(f,g,tempSizeOfSearchRegion,gridx,gridy,winsize,winstepsize,showWaitbar,ClusterNo)

if nargin < 8, showWaitbar = true; end
if nargin < 9, ClusterNo = 0; end
if length(winstepsize)==1, winstepsize = repmat(winstepsize,1,2); end
if length(winsize)==1, winsize = repmat(winsize,1,2); end

XList = [gridx(1) : winstepsize(1) : gridx(2)-winstepsize(1)];
YList = [gridy(1) : winstepsize(2) : gridy(2)-winstepsize(2)];
[XX,YY] = ndgrid(XList,YList);
temparrayLength = length(XList)*length(YList);
PtPosSeq = zeros(temparrayLength,2);
PtPosSeq(:,1) = XX(:); PtPosSeq(:,2) = YY(:);

utemp = zeros(temparrayLength,1); vtemp = utemp;
xtemp = utemp; ytemp = utemp; Phitemp = utemp;
qfactors = zeros(temparrayLength, 2);

nXList = length(XList);
x = zeros(length(YList),length(XList)); y = x; u = x; v = x; Phi = x;

%% ========== Start initial integer search ==========
if ClusterNo > 1
    % ------ Parallel computing ------
    if showWaitbar, hbar = parfor_progressbar(temparrayLength,'FFT initial guess (parallel)...'); end
    parfor tempi = 1:temparrayLength
        jj = PtPosSeq(tempi,2);
        ii = PtPosSeq(tempi,1);
        C = f(ii:ii+winsize(1), jj:jj+winsize(2));
        D = g(ii-tempSizeOfSearchRegion(1):ii+winsize(1)+tempSizeOfSearchRegion(1), ...
            jj-tempSizeOfSearchRegion(2):jj+winsize(2)+tempSizeOfSearchRegion(2));
        try
            XCORRF2OfCD0 = normxcorr2(C,D);
            ccLocal = struct('A', {{real(XCORRF2OfCD0)}});
            qfactors(tempi,:) = compute_qFactor(ccLocal, tempi);
            [v1temp, u1temp, max_f] = findpeak(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1),1);
            zero_disp = ceil(size(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1))/2);
            utemp(tempi) = u1temp-zero_disp(1);
            vtemp(tempi) = v1temp-zero_disp(2);
            Phitemp(tempi) = max_f;
        catch
            utemp(tempi)=nan; vtemp(tempi)=nan; Phitemp(tempi)=nan;
            qfactors(tempi,:) = [Inf,Inf];
        end
        ytemp(tempi) = (jj+jj+winsize(2))/2;
        xtemp(tempi) = (ii+ii+winsize(1))/2;
        if showWaitbar, hbar.iterate(1); end
    end
    if showWaitbar, close(hbar); end

else
    % ------ Serial computing ------
    if showWaitbar, hbar = waitbar(0,'FFT initial guess, please wait...'); end
    for tempi = 1:temparrayLength
        if showWaitbar, waitbar(tempi/temparrayLength); end
        jj = PtPosSeq(tempi,2);
        ii = PtPosSeq(tempi,1);
        C = f(ii:ii+winsize(1), jj:jj+winsize(2));
        D = g(ii-tempSizeOfSearchRegion(1):ii+winsize(1)+tempSizeOfSearchRegion(1), ...
            jj-tempSizeOfSearchRegion(2):jj+winsize(2)+tempSizeOfSearchRegion(2));
        try
            XCORRF2OfCD0 = normxcorr2(C,D);
            ccLocal = struct('A', {{real(XCORRF2OfCD0)}});
            qfactors(tempi,:) = compute_qFactor(ccLocal, tempi);
            [v1temp, u1temp, max_f] = findpeak(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1),1);
            zero_disp = ceil(size(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1))/2);
            utemp(tempi) = u1temp-zero_disp(1);
            vtemp(tempi) = v1temp-zero_disp(2);
            Phitemp(tempi) = max_f;
        catch
            utemp(tempi)=nan; vtemp(tempi)=nan; Phitemp(tempi)=nan;
            qfactors(tempi,:) = [Inf,Inf];
        end
        ytemp(tempi) = (jj+jj+winsize(2))/2;
        xtemp(tempi) = (ii+ii+winsize(1))/2;
    end
    if showWaitbar, close(hbar); end
end

%% Compute grid indices and assemble results
cj1temp = ceil((1:temparrayLength)' / nXList);
ci1temp = (1:temparrayLength)' - (cj1temp-1) * nXList;

for k = 1:2
    qf_ = (qfactors(:,k)-min(qfactors(:,k)));
    cc.qfactors(:,k) = qf_/max(qf_);
end

for tempi = 1:temparrayLength
    ci1 = ci1temp(tempi); cj1 = cj1temp(tempi);
    u(cj1,ci1) = utemp(tempi);
    v(cj1,ci1) = vtemp(tempi);
    Phi(cj1,ci1) = Phitemp(tempi);
    x(cj1,ci1) = xtemp(tempi);
    y(cj1,ci1) = ytemp(tempi);
end
disp('Finish initial guess search!');

cc.max = Phi; cc.A = [];


end




%% ==============================================

%%
function qfactors = compute_qFactor(cc,qnum)

%get peak locations and cc_min maps (i.e. cc - cc(min))
[peak,cc_min] = cellfun(@(x) cc_max_find(double(x)),cc.A,'UniformOutput',0);

%compute two primary quality metrics, as given in "Xue Z, Particle Image
% Velocimetry Correlation Signal-to-noise Metrics, Particle Image
% Pattern Mutual Information and Measurement uncertainty Quantification.
% MS Thesis, Virginia Tech, 2014.

%peak to corr. energy ratio
pce = cellfun(@(x,y) (abs(y)^2)/(1/numel(x)*(sum(abs(x(:)).^2))),cc_min,peak,'UniformOutput',0);
%min value -> 1 (worst case)

%peak to entropy ratio
ppe = cellfun(@(x) q_entropy(double(x)),cc_min,'UniformOutput',0);%peak to cc (information) entropy
%min value -> 0 (worst case)

qfactors = cell2mat(...
    cellfun(@(x,y) [x(:);y(:)], pce,ppe,'UniformOutput',0))';

end

function [peak,cc_min] = cc_max_find(cc)
%find the peak and zero-adjusted cc map

cc_min = cc - min(cc(:));%zero-adjust
% cc_filt = imgaussfilt3(cc_min); %filter to remove noise from peak value

[peak,~] = max(cc_min(:)); %get the index of the peak


end

function [ppe] = q_entropy(cc_min)
%compute entropy q-factor for a given cc map

[cc_hist,~] = histcounts(cc_min,30); %get histogram values

entropy = 0;
p = cc_hist/sum(cc_hist); %compute probablities
for i = 1:length(p)%compute entropy
    if p(i) == 0
        entropy = entropy+p(i);
    else
        entropy = entropy+p(i)*log(1/p(i));
    end
end

ppe = 1/entropy; %peak to cc (information) entropy
%min value -> 0 (worst case)


end









