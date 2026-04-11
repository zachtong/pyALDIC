function [f,dfdx,dfdy] = rbfinterp(x, options)
tic;
phi       = options.('rbfphi');
Dphi      = options.('rbfDphi');
rbfconst  = options.('RBFConstant');
nodes     = options.('x');
rbfcoeff  = (options.('rbfcoeff'))';


[dim              n] = size(nodes);
[dimPoints  nPoints] = size(x);

if (dim~=dimPoints)
  error(sprintf('x should have the same number of rows as an array used to create RBF interpolation'));
end;

f = zeros(1, nPoints);  dfdx=f; dfdy=f;
r = zeros(1, n);

for i=1:1:nPoints
	s=0; sx=0; sy=0;
    r = (x(:,i)*ones(1,n)) - nodes;
    r = sqrt(sum(r.*r, 1)) ;
%     for j=1:n
%          r(j) =  norm(x(:,i) - nodes(:,j));
%     end
    
     s = rbfcoeff(n+1) + sum(rbfcoeff(1:n).*feval(phi, r, rbfconst));
     sx = sum(rbfcoeff(1:n).*feval(Dphi, r(1:end), rbfconst) .* (x(1,i)*ones(1,n) - nodes(1,1:end)) ./ (r(1:end) +1e-6)  ) ;
     sy = sum(rbfcoeff(1:n).*feval(Dphi, r(1:end), rbfconst) .* (x(2,i)*ones(1,n) - nodes(2,1:end)) ./ (r(1:end) +1e-6)  ) ;
 
	for k=1:dim
       s=s+rbfcoeff(k+n+1)*x(k,i);     % linear part
    end
    sx = sx + rbfcoeff(1+n+1);  
    sy = sy + rbfcoeff(2+n+1); 
       
	f(i) = s;
    
    dfdx(i) = sx; dfdy(i) = sy; 
     
end;

if (strcmp(options.('Stats'),'on'))
    fprintf('Interpolation at %d points was computed in %e sec\n', length(f), toc);    
end;
