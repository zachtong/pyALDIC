function maxdiff = rbfcheck(options)

tic;

nodes     = options.('x');
    y     = options.('y');

s = rbfinterp(nodes, options);

maxdiff = max(abs(s-y));

% fprintf('RBF Check\n');
% fprintf('max|y - yi| = %e \n', max(abs(s-y)) );

if (strcmp(options.('Stats'),'on'))
    fprintf('%d points were checked in %e sec\n', length(y), toc);    
end

% fprintf('\n');
