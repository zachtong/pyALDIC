%% Integer Search

function [xGrid,yGrid,uGrid,vGrid,cc] = integer_search_mg(f,g,gridx,gridy,winsize,winstepsize,varargin)

% Parse optional showWaitbar flag (last varargin if logical)
showWaitbar = true;
if ~isempty(varargin) && islogical(varargin{end})
    showWaitbar = varargin{end};
    varargin(end) = [];
end

gridxBackup = gridx; gridyBackup = gridy; % may lose boundary regions, backup first
borderGapx = 1+round(0.75*winsize); borderGapy = borderGapx;
if gridx(1) < borderGapx, gridx(1) = borderGapx; end
if gridy(1) < borderGapy, gridy(1) = borderGapy; end
if gridx(2) > size(f,1)-borderGapx, gridx(2) = size(f,1)-borderGapx; end
if gridy(2) > size(f,2)-borderGapy, gridy(2) = size(f,2)-borderGapy; end
    
% f=ImgNormalized{1}; g=ImgNormalized{2}; % (10:end,20:end)  ImgNormalized{1}(1:end-9,1:end-19); %
% gridx=gridxROIRange; gridy=gridyROIRange;

%% Level 1
% First level cross-correlation
C = f(gridx(1):gridx(2),gridy(1):gridy(2));
D = g(gridx(1):gridx(2),gridy(1):gridy(2));
XCORRF2OfCD0 = normxcorr2(C,D); % figure, surf(XCORRF2OfCD0,'edgecolor','none')

% find maximum index of the cross-correlaiton
[v1temp, u1temp, max_f] = findpeak(XCORRF2OfCD0,1);
zero_disp = ceil((size(XCORRF2OfCD0)+[1,1])/2);

utemp = (u1temp-zero_disp(1)); % u(cj1,ci1)   = u1temp-zero_disp(1);
vtemp = (v1temp-zero_disp(2)); % v(cj1,ci1)   = v1temp-zero_disp(2);
Phitemp = max_f; % Phi(cj1,ci1) = max_f;

% Check out of image border or not
gridx_f = gridx; gridy_f = gridy;
gridx_g = gridx+[ceil(utemp),ceil(utemp)]; gridy_g = gridy+[ceil(vtemp),ceil(vtemp)]; %updated ROI in g, could be out of image
if gridx_g(1) < borderGapx, temp=gridx_g(1); gridx_g(1) = borderGapx; gridx_f(1) = gridx_f(1)+borderGapx-temp; end
if gridy_g(1) < borderGapy, temp=gridy_g(1); gridy_g(1) = borderGapy; gridy_f(1) = gridy_f(1)+borderGapy-temp; end
if gridx_g(2) > size(f,1)-borderGapx, temp=gridx_g(2); gridx_g(2) = size(f,1)-borderGapx; gridx_f(2) = gridx_f(2)+((size(f,1)-borderGapx)-temp); end
if gridy_g(2) > size(f,2)-borderGapy, temp=gridy_g(2); gridy_g(2) = size(f,2)-borderGapy; gridy_f(2) = gridy_f(2)+(size(f,2)-borderGapy)-temp; end
% Make sure image width/length odd number
if mod((gridx_f(2)-gridx_f(1)),2)==1, gridx_f(2)=gridx_f(2)-1; gridx_g(2)=gridx_g(2)-1; end
if mod((gridy_f(2)-gridy_f(1)),2)==1, gridy_f(2)=gridy_f(2)-1; gridy_g(2)=gridy_g(2)-1; end

% Visualize tracking
% figure,
% subplot(2,2,1); surf(f(gridxBackup(1):gridxBackup(2), gridyBackup(1):gridyBackup(2))','edgecolor','none'); view(2);axis equal;ylabel('y'); xlabel('x'); title('Raw f');
% subplot(2,2,2); surf(g(gridxBackup(1):gridxBackup(2), gridyBackup(1):gridyBackup(2))','edgecolor','none'); view(2);axis equal; ylabel('y'); xlabel('x'); title('Raw g');
% subplot(2,2,3); surf(f(gridx_f(1):gridx_f(2), gridy_f(1):gridy_f(2))','edgecolor','none'); view(2); axis equal;ylabel('y'); xlabel('x'); title('Shifted f');
% subplot(2,2,4); surf(g(gridx_g(1):gridx_g(2), gridy_g(1):gridy_g(2))','edgecolor','none'); view(2); axis equal;ylabel('y'); xlabel('x'); title('Shifted g');

gridxWidth0 = gridx_f(2)-gridx_f(1); gridyWidth0 = gridy_f(2)-gridy_f(1);
gridx_f0 = gridx_f; gridy_f0 = gridy_f; gridx_g0 = gridx_g; gridy_g0 = gridy_g;

%% ====== Assign values ======
gridxWidthCurr = gridxWidth0; gridyWidthCurr = gridyWidth0;
gridxyRatioCurr = gridxWidthCurr/gridyWidthCurr;
gridx_fCurr = gridx_f0; gridy_fCurr = gridy_f0; gridx_gCurr = gridx_g0; gridy_gCurr = gridy_g0;
utempCurr = ceil(utemp); vtempCurr = ceil(vtemp);

%% Level >1
levelNo=1; clear gridx_fNew gridx_gNew gridy_fNew gridy_gNew utempNew vtempNew qfactors
TotalNo = 1; gridxWidthNewtemp = gridxWidthCurr; gridyWidthNewtemp = gridyWidthCurr;
while gridxWidthNewtemp/gridyWidthNewtemp > 0
   if gridxWidthNewtemp/gridyWidthNewtemp > 2 % split gridx only 
       gridxWidthNewtemp = gridxWidthNewtemp/2; gridyWidthNewtemp = gridyWidthNewtemp; 
       TotalNo = TotalNo*2;
   elseif gridxWidthNewtemp/gridyWidthNewtemp < 0.5 % split gridy only
       gridxWidthNewtemp = gridxWidthNewtemp; gridyWidthNewtemp = gridyWidthNewtemp/2;
       TotalNo = TotalNo*2;
   else
       gridxWidthNewtemp = gridxWidthNewtemp/2; gridyWidthNewtemp = gridyWidthNewtemp/2;
       TotalNo = TotalNo*4;
   end
   if ( gridxWidthNewtemp<winsize && gridyWidthNewtemp<winsize )
        break
   end
end

%TotalNo = 3*(gridx(2)-gridx(1))*(gridy(2)-gridy(1))/winsize^2; 
IterNo=0; if showWaitbar, hbar = waitbar(0,'FFT initial guess, it is fast and please wait.'); end
while gridxyRatioCurr > 0
    levelNo=levelNo+1;
    clear utempNew vtempNew Phitemp
    % ====== Split gridx & gridy ======
    if gridxyRatioCurr > 2 % split gridx only
        gridxWidthNew = gridxWidthCurr/2; gridyWidthNew = gridyWidthCurr;
        for tempi = 1:size(gridx_fCurr,1)
            tempj=tempi; tempInd = tempj ;
            if ( (gridx_fCurr(tempInd,2)-gridx_fCurr(tempInd,1) > winsize)   )
                
                gridx_fNew(2*tempInd-1:2*tempInd,1:2) = [ gridx_fCurr(tempInd,1), 0.5*sum(gridx_fCurr(tempInd,:));
                    0.5*sum(gridx_fCurr(tempInd,:)), gridx_fCurr(tempInd,2)];
                gridx_gNew(2*tempInd-1:2*tempInd,1:2) = [ gridx_gCurr(tempInd,1), 0.5*sum(gridx_gCurr(tempInd,:));
                    0.5*sum(gridx_gCurr(tempInd,:)), gridx_gCurr(tempInd,2)];
                gridy_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_fCurr(tempInd,1:2), 2,1);
                gridy_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_gCurr(tempInd,1:2), 2,1);
                utempNew(2*tempInd-1:2*tempInd) = repmat(utempCurr(tempInd),2,1);
                vtempNew(2*tempInd-1:2*tempInd) = repmat(vtempCurr(tempInd),2,1);
            else
                gridx_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_fCurr(tempInd,1:2), 2,1);
                gridx_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_gCurr(tempInd,1:2), 2,1);
                gridy_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_fCurr(tempInd,1:2), 2,1);
                gridy_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_gCurr(tempInd,1:2), 2,1);
                utempNew(2*tempInd-1:2*tempInd) = repmat(utempCurr(tempInd),2,1);
                vtempNew(2*tempInd-1:2*tempInd) = repmat(vtempCurr(tempInd),2,1);
            end
            
        end
        
    elseif gridxyRatioCurr < 0.5 % split gridy only
        gridxWidthNew = gridxWidthCurr; gridyWidthNew = gridyWidthCurr/2;
        for tempi = 1:size(gridx_fCurr,1)
            tempj=tempi; tempInd = tempj ;
            if (  (gridy_fCurr(tempInd,2)-gridy_fCurr(tempInd,1) > winsize) )
                
                gridx_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_fCurr(tempInd,1:2), 2,1);
                gridx_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_gCurr(tempInd,1:2), 2,1);
                gridy_fNew(2*tempInd-1:2*tempInd,1:2) = [ gridy_fCurr(tempInd,1), 0.5*sum(gridy_fCurr(tempInd,:));
                    0.5*sum(gridy_fCurr(tempInd,:)), gridy_fCurr(tempInd,2)];
                gridy_gNew(2*tempInd-1:2*tempInd,1:2) = [ gridy_gCurr(tempInd,1), 0.5*sum(gridy_gCurr(tempInd,:));
                    0.5*sum(gridy_gCurr(tempInd,:)), gridy_gCurr(tempInd,2)];
                utempNew(2*tempInd-1:2*tempInd) = repmat(utempCurr(tempInd),2,1);
                vtempNew(2*tempInd-1:2*tempInd) = repmat(vtempCurr(tempInd),2,1);
            else
                gridx_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_fCurr(tempInd,1:2), 2,1);
                gridx_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridx_gCurr(tempInd,1:2), 2,1);
                gridy_fNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_fCurr(tempInd,1:2), 2,1);
                gridy_gNew(2*tempInd-1:2*tempInd,1:2) = repmat( gridy_gCurr(tempInd,1:2), 2,1);
                utempNew(2*tempInd-1:2*tempInd) = repmat(utempCurr(tempInd),2,1);
                vtempNew(2*tempInd-1:2*tempInd) = repmat(vtempCurr(tempInd),2,1);
            end
             
        end
        
    else % split both in gridx and griy directions
        gridxWidthNew = gridxWidthCurr/2; gridyWidthNew = gridyWidthCurr/2;
        for tempi = 1:size(gridx_fCurr,1)
            tempj=tempi; tempInd = tempj ;
            if ( (gridx_fCurr(tempInd,2)-gridx_fCurr(tempInd,1) > winsize) && (gridy_fCurr(tempInd,2)-gridy_fCurr(tempInd,1) > winsize) )
                gridx_fNew(4*tempInd-3:4*tempInd,1:2) = repmat([ gridx_fCurr(tempInd,1), 0.5*sum(gridx_fCurr(tempInd,:));
                    0.5*sum(gridx_fCurr(tempInd,:)), gridx_fCurr(tempInd,2)], 2,1);
                gridx_gNew(4*tempInd-3:4*tempInd,1:2) = repmat([ gridx_gCurr(tempInd,1), 0.5*sum(gridx_gCurr(tempInd,:));
                    0.5*sum(gridx_gCurr(tempInd,:)), gridx_gCurr(tempInd,2)], 2,1);
                gridy_fNew(4*tempInd-3:4*tempInd,1:2) = [ repmat([ gridy_fCurr(tempInd,1), 0.5*sum(gridy_fCurr(tempInd,:))],2,1);
                    repmat([0.5*sum(gridy_fCurr(tempInd,:)), gridy_fCurr(tempInd,2)],2,1) ];
                gridy_gNew(4*tempInd-3:4*tempInd,1:2) =  [ repmat([ gridy_gCurr(tempInd,1), 0.5*sum(gridy_gCurr(tempInd,:))],2,1);
                    repmat([0.5*sum(gridy_gCurr(tempInd,:)), gridy_gCurr(tempInd,2)],2,1) ];
                utempNew(4*tempInd-3:4*tempInd) = repmat(utempCurr(tempInd),4,1);
                vtempNew(4*tempInd-3:4*tempInd) = repmat(vtempCurr(tempInd),4,1);
            else
                gridx_fNew(4*tempInd-3:4*tempInd,1:2) = repmat( gridx_fCurr(tempInd,1:2), 4,1);
                gridx_gNew(4*tempInd-3:4*tempInd,1:2) = repmat( gridx_gCurr(tempInd,1:2), 4,1);
                gridy_fNew(4*tempInd-3:4*tempInd,1:2) = repmat( gridy_fCurr(tempInd,1:2), 4,1);
                gridy_gNew(4*tempInd-3:4*tempInd,1:2) = repmat( gridy_gCurr(tempInd,1:2), 4,1);
                utempNew(4*tempInd-3:4*tempInd) = repmat(utempCurr(tempInd),4,1);
                vtempNew(4*tempInd-3:4*tempInd) = repmat(vtempCurr(tempInd),4,1);
            end
            
        end
        
    end
    
    
    for tempi = 1:size(gridx_fNew,1)
        IterNo = IterNo+1;
        if showWaitbar, waitbar(IterNo/TotalNo); end
        
        try
            C = f(gridx_fNew(tempi,1):gridx_fNew(tempi,2), gridy_fNew(tempi,1):gridy_fNew(tempi,2));
            D = g(gridx_gNew(tempi,1)-5:gridx_gNew(tempi,2)+5, gridy_gNew(tempi,1)-5:gridy_gNew(tempi,2)+5);
            winsize1 = size(C,1)-1; winsize2 = size(C,2)-1;

        
            XCORRF2OfCD0 = normxcorr2(C,D); % cross-correlation
            % find qfactors
            cc.A{1} = real(XCORRF2OfCD0);
            qfactors(tempi,:) = compute_qFactor(cc,tempi);

            [v1temp,u1temp,max_f] = findpeak(XCORRF2OfCD0(winsize1:end-winsize1+1, winsize2:end-winsize2+2),1);
            zero_disp = ceil(size(XCORRF2OfCD0(winsize1:end-winsize1+1,winsize2:end-winsize2+1))/2);

            ind = tempi;

            utempNew(ind) = utempNew(ind) + (u1temp-zero_disp(1)) ;
            vtempNew(ind) = vtempNew(ind) + (v1temp-zero_disp(2));
            Phitemp(ind) = max_f;
        catch
            ind = tempi;
            cc.A{1} = nan;  qfactors(tempi,:) = nan; 
            utempNew(ind) = nan;  vtempNew(ind) = nan;  Phitemp(ind) = 0;
        end
        
    end
    
    if ( gridxWidthNew<winsize && gridyWidthNew<winsize )
        % Finish qfactor computation
        for k=1:2
            qf_ = (qfactors(:,k)-min(qfactors(:,k)));
            cc.qfactors(:,k) = qf_/max(qf_);
        end
        break
    else
        
        %% ====== Check out of image border or not ======
        gridx_f = gridx_fNew; gridy_f = gridy_fNew;
        gridx_g = gridx_fNew + ceil(utempNew)'*[1,1];
        gridy_g = gridy_fNew + ceil(vtempNew)'*[1,1];
        for tempi = 1:size(gridx_g,1)
            if gridx_g(tempi,1)<borderGapx
                temp=gridx_g(tempi,1);
                gridx_g(tempi,1)=borderGapx;
                gridx_f(tempi,1) = gridx_f(tempi,1)+borderGapx-temp;
            end
            if gridy_g(tempi,1)<borderGapy
                temp=gridy_g(tempi,1);
                gridy_g(tempi,1)=borderGapy;
                gridy_f(tempi,1) = gridy_f(tempi,1)+borderGapy-temp;
            end
            if gridx_g(tempi,2)>size(f,1)-borderGapx
                temp=gridx_g(tempi,2);
                gridx_g(tempi,2)=size(f,1)-borderGapx;
                gridx_f(tempi,2) = gridx_f(tempi,2)+(size(f,1)-borderGapx)-temp;
            end
            if gridy_g(tempi,2)>size(f,2)-borderGapy
                temp=gridy_g(tempi,2);
                gridy_g(tempi,2)=size(f,2)-borderGapy;
                gridy_f(tempi,2) = gridy_f(tempi,2)+(size(f,2)-borderGapy)-temp;
            end
        end
        % Make sure: image width/length odd number
        for tempi = 1:size(gridx_g,1)
            if mod(gridx_f(tempi,2)-gridx_f(tempi,1),2)==1
                gridx_f(tempi,2)=gridx_f(tempi,2)-1;
                gridx_g(tempi,2)=gridx_g(tempi,2)-1;
            end
            if mod(gridy_f(tempi,2)-gridy_f(tempi,1),2)==1
                gridy_f(tempi,2)=gridy_f(tempi,2)-1;
                gridy_g(tempi,2)=gridy_g(tempi,2)-1;
            end
        end
        
        %% ====== Assign values ======
        gridxWidthCurr = gridxWidthNew; gridyWidthCurr = gridyWidthNew;
        gridxyRatioCurr = gridxWidthCurr/gridyWidthCurr;
        gridx_fCurr = gridx_f; gridy_fCurr = gridy_f; gridx_gCurr = gridx_g; gridy_gCurr = gridy_g;
        utempCurr =  ceil(utempNew); vtempCurr =  ceil(vtempNew);
        
    end
    
end

if showWaitbar, close(hbar); end


%%
tempx=0.5*(gridx_fNew(:,1)+gridx_fNew(:,2));
tempy = 0.5*(gridy_fNew(:,1)+gridy_fNew(:,2));
tempu=utempNew'; tempv=vtempNew'; tempPhi=Phitemp';
qf_1 = cc.qfactors(:,1)'; qf_2 = cc.qfactors(:,2)';
% figure, plot3(tempx,tempy,tempu,'.');
% figure, plot3(tempx,tempy,tempv,'.');

%% Update image domain
[~,indx1]=min(tempx); [~,indx2]=max(tempx);
[~,indy1]=min(tempy); [~,indy2]=max(tempy);
borderGapx1=ceil(1+(1.25*winsize)+((tempu(indx1)))); borderGapx2=ceil(1+(1.25*winsize)+((tempu(indx2)))); 
borderGapy1=ceil(1+(1.25*winsize)+((tempv(indy1)))); borderGapy2=ceil(1+(1.25*winsize)+((tempv(indy2))));
if gridx(1) < borderGapx1, gridx(1) = borderGapx1; end
if gridy(1) < borderGapy1, gridy(1) = borderGapy1; end
if gridx(2) > size(f,1)-borderGapx2+1, gridx(2) = size(f,1)-borderGapx2+1; end
if gridy(2) > size(f,2)-borderGapy2+1, gridy(2) = size(f,2)-borderGapy2+1; end

%% Interpolate
try xList=[gridx(1):winstepsize(1):gridx(2)]; catch, xList=[gridx(1):winstepsize:gridx(2)]; end
try yList=[gridy(1):winstepsize(2):gridy(2)]; catch, yList=[gridy(1):winstepsize:gridy(2)]; end
[indx]=find(tempx>min(xList) & tempx<max(xList));
[indy]=find(tempy>min(yList) & tempy<max(yList));
[indxy] = intersect(indx,indy);
[uGrid,xGrid,yGrid] = gridfit(tempx(indxy),tempy(indxy),tempu(indxy),xList,yList,'regularizer','springs');
[vGrid,~,~] = gridfit(tempx(indxy),tempy(indxy),tempv(indxy),xList,yList,'regularizer','springs');
[PhiGrid,~,~] = gridfit(tempx(indxy),tempy(indxy),tempPhi(indxy),xList,yList,'regularizer','springs');
 
[qf_1Grid,~,~] = gridfit(tempx(indxy),tempy(indxy),qf_1(indxy),xList,yList,'regularizer','springs');
[qf_2Grid,~,~] = gridfit(tempx(indxy),tempy(indxy),qf_2(indxy),xList,yList,'regularizer','springs');
cc.max = PhiGrid; cc.A = []; cc.qfactors =[qf_1Grid(:),qf_2Grid(:)]; 
% figure, surf(xGrid,yGrid,uGrid,'edgecolor','none')
% figure, surf(xGrid,yGrid,vGrid,'edgecolor','none')
 
end

function [x,y,u,v,cc] = funIntegerSearchWholeField(f,g,tempSizeOfSearchRegion,gridx,gridy,winsize,winstepsize,showWaitbar)

%cj1 = 1; ci1 = 1; % index to count main loop

if length(winstepsize)==1
    winstepsize = repmat(winstepsize,1,2);
end
if length(winsize)==1
    winsize = repmat(winsize,1,2);
end

% disp('Assemble point position sequence.');
XList = [gridx(1) : winstepsize(1) : gridx(2)-winstepsize(1)];
YList = [gridy(1) : winstepsize(2) : gridy(2)-winstepsize(2)];
[XX,YY] = ndgrid(XList,YList);
temparrayLength = length(XList)*length(YList);
PtPosSeq = zeros(temparrayLength,2);
PtPosSeq(:,1) = XX(:); PtPosSeq(:,2) = YY(:);

cj1temp = zeros(temparrayLength,1); ci1temp = cj1temp;
utemp = cj1temp; vtemp = cj1temp;
xtemp = cj1temp; ytemp = cj1temp; Phitemp = cj1temp;

%% ========== Start initial integer search ==========
if nargin < 8, showWaitbar = true; end
if showWaitbar, hbar = waitbar(0,'FFT initial guess, it is fast and please wait.'); end
% hbar = parfor_progressbar(temparrayLength,'Please wait for integer search!');

% sizeOfx1 = floor((gridx(2)-gridx(1)-winsize)/winstepsize)+1;
% sizeOfx2 = floor((gridy(2)-gridy(1)-winsize)/winstepsize)+1;
% disp(['Init search using FFT cross correlation on grid: ',num2str(sizeOfx1),'x',num2str(sizeOfx2)]);
x = zeros(length(YList),length(XList)); y = x; u = x; v = x; Phi = x;

for tempi = 1:temparrayLength
    
    if showWaitbar, waitbar(tempi/temparrayLength); end

    jj = PtPosSeq(tempi,2); % for jj = gridy(1) : winstepsize : gridy(end)-winsize
    % jj is for y -or- vertical direction of images
    
    ii = PtPosSeq(tempi,1);%for ii = gridx(1) : winstepsize : gridx(end)-winsize
    % ii is for x -or- horizontal direction of images
    
    C = f(ii:ii+winsize(1), jj:jj+winsize(2));
    
    D = g(ii-tempSizeOfSearchRegion:ii+winsize(1)+tempSizeOfSearchRegion, ...
        jj-tempSizeOfSearchRegion:jj+winsize(2)+tempSizeOfSearchRegion);
    
    XCORRF2OfCD0 = normxcorr2(C,D);
    
    %find qfactors
    cc.A{1} = real(XCORRF2OfCD0);
    qfactors(tempi,:) = compute_qFactor(cc,tempi);
    
    % find maximum index of the cross-correlaiton
    [v1temp, u1temp, max_f] = findpeak(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1),1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % I tried following code, unfortunately it doesn't work very well
    % [v1temp1, u1temp1, max_f1] = findpeak(XCORRF2OfCD0(winsize:end-winsize+1,winsize:end-winsize+1),1);
    % [v1temp2, u1temp2, max_f2] = findpeak(XCORRF2OfCD0(winsize:end-winsize+1,winsize:end-winsize+1),0);
    %
    % if max_f2 > 0.999
    %    v1temp = v1temp2; u1temp = u1temp2; max_f = max_f2;
    % else
    %    v1temp = v1temp1; u1temp = u1temp1; max_f = max_f1;
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    zero_disp = ceil(size(XCORRF2OfCD0(winsize(1):end-winsize(1)+1,winsize(2):end-winsize(2)+1))/2);
    
    utemp(tempi) = u1temp-zero_disp(1); % u(cj1,ci1)   = u1temp-zero_disp(1);
    vtemp(tempi) = v1temp-zero_disp(2); % v(cj1,ci1)   = v1temp-zero_disp(2);
    Phitemp(tempi) = max_f; % Phi(cj1,ci1) = max_f;
    
    ytemp(tempi) = (jj+jj+winsize(2))/2; % y(cj1,ci1)=(jj+jj+winsize)/2;   % vertical position in image
    xtemp(tempi) = (ii+ii+winsize(1))/2; % x(cj1,ci1)=(ii+ii+winsize)/2;   % horizontal position in image
    
    % Update counters
    %ci1 = ci1 + 1;  % ci1 is moving horizontally for subsets
    
    %end
    
    %ci1=1; cj1=cj1+1;  % cj1 is moving vertically for subsets
    cj1temp(tempi) = ceil(tempi/length(XList));
    ci1temp(tempi) = tempi - (cj1temp(tempi)-1) * length(XList)  ;
    
end

if showWaitbar, close(hbar); end

for k = 1:2
    qf_ = (qfactors(:,k)-min(qfactors(:,k)));
    cc.qfactors(:,k) = qf_/max(qf_);
end

% hbar = parfor_progressbar(temparrayLegTotal,'Assign results to variables.');
if showWaitbar, hbar = waitbar(0,'Assign results to variables.'); end
for tempi = 1:temparrayLength
    
    ci1 = ci1temp(tempi); cj1 = cj1temp(tempi);
    
    u(cj1,ci1) = utemp(tempi);
    v(cj1,ci1) = vtemp(tempi);
    
    Phi(cj1,ci1) = Phitemp(tempi);
    
    x(cj1,ci1) = xtemp(tempi);
    y(cj1,ci1) = ytemp(tempi);
    
    if showWaitbar, waitbar(tempi/temparrayLength); end
    % hbar.iterate(1);
end
if showWaitbar, close(hbar); end
disp('Finish initial guess search!');
% -------- End of Local integer search --------


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









