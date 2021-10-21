function [w_hat,w1,w2,w3] = trilinearRegress_coordAscent(xx,xy,wDims,rnk,lambda,opts)
% [w_hat,what1,what2,what3] = trilinearRegress_coordAscent(xx,xy,wDims,rnk,lambda,opts)
% 
% Computes linear regression estimate with a rank-1 trilinear parametrization of the regression weights.  
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where w is parametrized as a rank-1 3rd order tensor: w1 ^ w2 ^ w3;
%
% Inputs:
% -------
%   xx - autocorrelation of design matrix (unnormalized)
%   xy - crosscorrelation between design matrix and dependent var (unnormalized)
%   wDims - [n1, n2, n3] 3-vector specifying # tensor coeffs for each dimension of the filter
%   p - rank of tensor (MUST BE 1 FOR NOW)
%   lambda - ridge parameter (optional)
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:
% -------
%   wHat = estimate of full param vector
%   wt = column vectors
%   wx = row vectors

if (nargin >= 5) && ~isempty(lambda)  
    xx = xx + lambda*eye(size(xx)); % add ridge penalty to xx
end

if (nargin < 6) || isempty(opts)
    opts.MaxIter = 25;
    opts.TolFun = 1e-6;
    opts.Display = 'iter';
end

% Set some params
n1 = wDims(1); n2 = wDims(2); n3 = wDims(3);
I1 = speye(n1); I2 = speye(n2); I3 = speye(n3); 

% % compute full rank least-squares solution
wLS = xx\xy; 

% --- Crude initialization: use SVDs to find low-rank initialization ----
% 1st svd for w1 vs others
[w1,s,w2] = svd(reshape(wLS,n1,n2*n3),'econ'); 
w1 = w1(:,1:rnk);
w2 = w2(:,1:rnk)*s(1:rnk,1:rnk);
% 2nd svd for w2 vs w3
[w2,s,w3] = svd(reshape(w2,n2,n3),'econ'); 
w2 = w2(:,1:rnk);
w3 = w3(:,1:rnk)*s(1:rnk,1:rnk);

% --- compute initial error and initialize coordinate ascent --------
w = vec(mkrank1tensor(w1,w2,w3));
fval = .5*w'*xx*w - w'*xy;
fchange = inf;
iter = 1;
if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of trilinear loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

% --- Start coordinate ascent ----------------------------------------

while (iter <= opts.MaxIter) && (fchange > opts.TolFun)

    % Update w1 component
    M1 = kron(vec(w2*w3'),I1);
    w1 = (M1'*xx*M1)\(M1'*xy);
    w1 = reshape(w1, n1,rnk);
    
    % Update w2 component
    M2 = kron(w3,kron(I2,w1));
    w2 = (M2'*xx*M2)\(M2'*xy);
    w2 = reshape(w2,n2,rnk);
    
    % Update spatial components
    M3 = kron(I3, vec(w1*w2'));
    w3 = (M3'*xx*M3)\(M3'*xy);
    w3 = reshape(w3,n3,rnk);

    % Compute size of change 
    w = vec(mkrank1tensor(w1,w2,w3));
    fvalnew = .5*w'*xx*w - w'*xy;
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(opts.Display, 'iter')
	fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

w_hat = mkrank1tensor(w1,w2,w3);