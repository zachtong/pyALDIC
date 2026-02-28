function [u_pred,alpha,beta_POD,beta_GPR] = por_gpr(T_snap,t_train,t_pre,nB)
% Input:
% T_snap = nT * np matrix; snapshots data
% t_train = nT * 1 vector; time for snapshots data
% t_pre = nT1 * vector; time to predict
% nB: number of basis; 
% Output:
% u_pred = nT1 * np matrix; Prediction matrix
% alpha, beta_POD, beta_POD = some nondimensional parameters


np = length(T_snap(1,:));
nT = length(T_snap(:,1));
nT1 = length(t_pre(:,1));

% POD 
% mean and fluctuation
ave = mean(T_snap);
fp  = zeros(nT,length(T_snap(1,:)));

for k=1:nT
    fp(k,:)=T_snap(k,:)-ave;
end

% correlation matrix
c = zeros(nT,nT);
for k=1:nT
    for j=1:nT
        c(k,j)=innerproduct(fp(k,:),fp(j,:));
    end
end

% POD decomposition
[w,d] = eig(c);
lambda = wrev(diag(d));
w = fliplr(w);
beta_POD = (log(lambda(2))-log(lambda(nB+2)))/nB;
sum1 = sum(lambda);
w = w(:,1:nB);
fi = fp'*w;
fi = fi';
lambda = lambda(1:nB);
sum2 = sum(lambda);
alpha = sum2/sum1;


% fi: basis
for j=1:nB
    fi(j,:)=fi(j,:)/sqrt(lambda(j));
end

% get a_exact
a_exact=zeros(nT,nB);
for j=1:nT
    for k=1:nB
        a_exact(j,k)=innerproduct(fp(j,:),fi(k,:));
    end
end

% GP regression
ypred=zeros(nT1,nB);
ysd=zeros(nT1,nB);
yint=zeros(nT1,nB*2);
for k=1:nB
%     beta=[1;1;0.01;0.001];
    gprMdl = fitrgp(t_train,a_exact(:,k),'Basis','linear','FitMethod','exact', ...
        'PredictMethod','exact','SigmaLowerBound',0.00002);

    [ypred(:,k), ysd(:,k), yint(:,2*k-1:2*k)] = predict(gprMdl,t_pre);
end


% compute prediction at t_pre
u_pred = ypred*fi+ave;

% compute beta_GPR
sd = ysd * lambda;
weighted_sum = abs(ypred) * lambda;
beta_GPR = sd./weighted_sum;



function innerproduct = innerproduct(f1,f2)
%   inner product of f1 and f2
N=length(f1)-1;
% x=0:1/N:1;
y=f1.*f2;
innerproduct=y(1)/2;
for i=2:N
    innerproduct=innerproduct+y(i);
end
innerproduct=0.1/N*(innerproduct+y(N+1)/2);
end






end