function [w1,w2,w3,wvec] = trilinearMultifiltRegress_coordAscent(xx,xy,n1,n2,n3,rnk,lambda,opts)
% [w1,w2,w3,wvec] = trilinearMultifiltRegress_coordAscent(xx,xy,n1,n2,n3,rnk,lambda,opts)
% 
% Computes regression estimate with multiple filters, each parametrized trilinearly
% Note: only handles rank 1 for now!
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where w has multiple partitions, each parametrized as a rank-1 tensor:
%  wfilt_i = w1_i ^ w2_i ^ w3_i
%
% Inputs:
% -------
%   xx - autocorrelation of design matrix (unnormalized)
%   xy - crosscorrelation between design matrix and dependent var (unnormalized)
%   n1 - # of rows of each weight tensor
%   n2 - # of columns of each weight tensor
%   n3 - # of elements in 3rd dimension of weight tensor
%   rnk - rank of trilinear filters  (must be 1, for now)
%   lambda - ridge parameter (optional)
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:
% -------
%   wstruct  = vector of structures with fields:
%     .w1 = column vectors 
%     .w2 = row vectors 
%     .w3 = 'depth' vectors 
%     .wfilt = w1 ^ w2 ^ w3;
%   wvec = full vector of regression weights

% ----------------------------------
% set optimization options
if (nargin < 8) || isempty(opts)
    opts.default = true;
end
if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

% ----------------------------------
% add ridge penalty to diagonal of xx
if (nargin >= 7) && ~isempty(lambda)  
    xx = xx + lambda*eye(size(xx)); 
end

% -----------------------------------
% Set some size params and build needed indices & sparse matrices
nfilts = length(n1); % number of filters
nw = zeros(nfilts,1); % number of coeffs in each filter
nw1 = zeros(nfilts,1); % number of low rank column vector params
nw2 = zeros(nfilts,1); % number of low rank row vector params
nw3 = zeros(nfilts,1); % number of low rank depth vector params
n1cll = num2cell(n1(:));   % length of col vectors (as cell array)
n2cll = num2cell(n2(:));   % length of row vectors (as cell array)
n3cll = num2cell(n3(:));   % length of depth vectors (as cell array)
rnkcll = num2cell(rnk(:)); % filter ranks (as cell array)

% Calculate sizes 
icum = 0; % counter 
inds = cell(nfilts,1);  % indices for each partition of the xy vector
I1 = cell(nfilts,1); I2 = cell(nfilts,1); I3 = cell(nfilts,1);
for jj = 1:nfilts
    nw(jj) = n1(jj)*n2(jj)*n3(jj);  % number of filter coeffs in each filter
    nw1(jj) = rnk(jj)*n1(jj); % number of params in column vectors
    nw2(jj) = rnk(jj)*n2(jj); % number of params in row vectors
    nw3(jj) = rnk(jj)*n3(jj); % number of params in depth vectors

    % set indices for these coeffs
    inds{jj} = icum+1:icum+nw(jj);
    icum = icum+nw(jj);
    
    % build needed sparse identity matrices
    I1{jj} = speye(n1(jj));
    I2{jj} = speye(n2(jj));
    I3{jj} = speye(n3(jj));    
end

% -----------------------------------
% Initialize estimate of w by linear regression and SVD
wLS = xx\xy;
w1=cell(nfilts,1); w2=cell(nfilts,1); w3=cell(nfilts,1); % initialize
% do SVD on each relevant portion of w0
for jj = 1:nfilts
    
    % first SVD to initialize w1
    [w10,s,w20] = svd(reshape(wLS(inds{jj}),n1(jj),n2(jj)*n3(jj)),'econ'); 
    w1{jj} = w10(:,1:rnk(jj));
    w20 = w20(:,1:rnk(jj))*s(1:rnk(jj),1:rnk(jj));

    % first SVD to initialize w2 and w3
    [w20,s,w30] = svd(reshape(w20,n2(jj),n3(jj)),'econ'); % 2nd svd for x1 vs x2
    w2{jj} = w20(:,1:rnk(jj));
    w3{jj} = w30(:,1:rnk(jj))*s(1:rnk(jj),1:rnk(jj));
end

% -----------------------------------
% Set up coordinate ascent

% define useful function: vectorize after matrix multiply
vecTensr = @(x1,x2,x3)(vec(mkrank1tensor(x1,x2,x3))); 

% Compute full weight vector from its low-rank components
wvec = cellfun(vecTensr,w1,w2,w3,'UniformOutput',false); % full weights in cell array
wvec = cell2mat(wvec); % convert to vector

% Evaluate loss function for initial weights
fval = .5*wvec'*xx*wvec - wvec'*xy;   % initial loss
fchange = inf; % initial value
iter = 1; % counter

if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of trilinear loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

% define three useful function for building Kronecker matrices
kronfun1 = @(v2,v3,M)(kron(vec(v2*v3'),M)); 
kronfun2 = @(v1,v3,M)(kron(v3,kron(M,v1)));
kronfun3 = @(v1,v2,M)(kron(M,vec(v1*v2')));

% -----------------------------------
% Run coordinate ascent

while (iter <= opts.MaxIter) && (fchange > opts.TolFun)
        
    % ===========================
    % Update w1 components
    M1 = cellfun(kronfun1,w2,w3,I1,'UniformOutput',false); % cell array of kronecker matrices
    M1 = blkdiag(M1{:});  % convert to block-diagonal matrix
    w1new = (M1'*xx*M1)\(M1'*xy);  % new temporal weights as column vector
    
    % Store new w1 weights in cell array
    w1new = mat2cell(w1new,nw1,1);  % break into cell array of vectors
    w1 = cellfun(@reshape,w1new,n1cll,rnkcll,'UniformOutput',false); % reshape into appropriate columns
    
    % ===========================
    % Update w2 components
    M2 = cellfun(kronfun2,w1,w3,I2,'UniformOutput',false); % cell array of kronecker matrices
    M2 = blkdiag(M2{:});  % convert to block-diagonal matrix
    w2new = (M2'*xx*M2)\(M2'*xy); % new spatial weights as column vector
    
    % Store new w2 weights in cell array    
    w2new = mat2cell(w2new,nw2,1);  % break into cell array of vectors
    w2 = cellfun(@reshape,w2new,n2cll,rnkcll,'UniformOutput',false); % reshape into appropriate columns
    
    % ===========================
    % Update w3 components
    M3 = cellfun(kronfun3,w1,w2,I3,'UniformOutput',false); % cell array of kronecker matrices
    M3 = blkdiag(M3{:});  % convert to block-diagonal matrix
    w3new = (M3'*xx*M3)\(M3'*xy); % new spatial weights as column vector
    
    % Store new w3 weights in cell array    
    w3new = mat2cell(w3new,nw3,1);  % break into cell array of vectors
    w3 = cellfun(@reshape,w3new,n3cll,rnkcll,'UniformOutput',false); % reshape into appropriate columns
    
    
    % Compute full weight vector from its low-rank components
    % Compute full weight vector from its low-rank components
    wvec = cellfun(vecTensr,w1,w2,w3,'UniformOutput',false); % full weights in cell array
    wvec = cell2mat(wvec); % convert to vector
    
    % Evaluate loss function & size of change
    fvalnew = .5*wvec'*xx*wvec - wvec'*xy;
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(opts.Display, 'iter')
        fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

