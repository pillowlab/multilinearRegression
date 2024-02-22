function [wU,wVt,wwfilts] = bilinearMultifiltRRR_coordAscent(Xin,Yout,nin,rnks,lambda,opts)
% [wU,wVt,wwfilts] = bilinearMultifiltRRR_coordAscent(X,Y,nin,rnks,lambda,opts)
% 
% Computes regression estimate with a bilinear parametrization of part of
% the parameter vector.
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where part of w is parametrized as vec(wt*wx')
%
% Inputs:
% -------
%   X - input population design matrices (T x nintot)
%   Y - output population response matrix (T x nout) 
%   nin - # of input neurons in each input populatin
%   rnks - rank of each low-rank filter
%   lambda - ridge parameter (optional)  (NOT YET SUPPORTED)
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:
% -------
%   wstruct  = vector of structures with fields:
%     .wt = column vectors 
%     .wx = row vectors 
%     .wfilt = wt*wx';
%   wvec = full vector of regression weights

% ----------------------------------
% set optimization options
if (nargin < 6) || isempty(opts)
    opts.default = true;
end
if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

ninpops = length(nin); % number of input populations
nintot = sum(nin);     % total number of input neurons
nout = size(Yout,2);

% -----------------------------------
% Initialize estimate of w by ridge regression and SVD
% (Note: an alternative would be to initialize with RRR)

% Compute sufficient statistics
Xfull = cell2mat(Xin);
XX = Xfull'*Xfull;  % input covariance
XY = Xfull'*Yout; % input-output cross-covariance

% compute ridge regression estimate
wridge = (XX+(lambda)*eye(nintot))\XY; % ridge regression solution 
wridgefilts = mat2cell(wridge,nin,nout); % convert into cell array for individual filters

% do SVD on each relevant portion of w0
wU = cell(ninpops,1);
wVt = cell(ninpops,1);
for jj = 1:ninpops
    [u0,s,v0] = svd(wridgefilts{jj},'econ'); % SVD
    ii = 1:rnks(jj); % indices of singular vectors to keep
    wU{jj} = u0(:,ii)*sqrt(s(ii,ii));  % left singular vectors
    wVt{jj} = sqrt(s(ii,ii))*v0(:,ii)'; % right singular vectors (weighted by singular values)
end

% -----------------------------------
% Initialize coordinate ascent

% % define useful function: vectorize after matrix multiply
% vecMatMult = @(x,y)(vec(mtimes(x,y))); 
% 
% % Compute full weight vector from its low-rank components
% wwfilts = cellfun(vecMatMult,wu,wv,'UniformOutput',false); % full weights in cell array
% wwfilts = cell2mat(wwfilts); % convert to vector

ww0 = cellfun(@mtimes,wU,wVt,'UniformOutput',false);
ww0 = cell2mat(ww0);

fval = 0.5*trace(ww0'*XX*ww0) - sum(sum((ww0.*XY)));
fchange = inf;
iter = 1;

if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of multi-filter RRR loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

% % define some useful cell functions and a cell array we'll need
fun_Ureshape = @(v,r)(reshape(v,[],r)); % function to reshape U matrices 
fun_mtimesTrp = @(A,B)(vec(mtimes(A,B')));       % function to compute kronecker of transpose with matrix
rnks_cell = num2cell(rnks(:));          % filter ranks as cell array
XXc = mat2cell(XX,nin,nin);
XYc = mat2cell(XY,nin,nout);
% -----------------------------------
% Run coordinate ascent

while (iter <= opts.MaxIter) && (fchange > opts.TolFun)
    
    % ===========================
    % Update U (column vectors)
        
    % project Xin onto corresponding V matrices and do regression
    
    % Compute Covariance (Vt' o Vt)(X o X)
    Vt = cell2mat(wVt);
    VtVc = mat2cell(Vt*Vt',rnks,rnks);
    MvtMv = cell2mat(cellfun(@kron,VtVc,XXc,'UniformOutput',false));
    
    % Compute Vectorized Cross-covariance (Vt' o X')Y
    MvtY = cell2mat(cellfun(fun_mtimesTrp,XYc,wVt,'UniformOutput',false));
        
    % solve regression problem for U
    Unew = (MvtMv)\MvtY; 
    
    % Place Unew into cell array
    wU = mat2cell(Unew,nin.*rnks,1); % convert to cell array
    wU = cellfun(fun_Ureshape,wU,rnks_cell,'UniformOutput',false); % reshape each filter to correct shape

    % ===========================
    % Update Vt (row vectors)

    % project Xin onto corresponding U matrices and do regression
    Mu = cell2mat(cellfun(@mtimes,Xin,wU','UniformOutput',false)); % design matrix for estimating Vt
    Vtnew = (Mu'*Mu)\(Mu'*Yout); % solve regression problem for Vt
    wVt = mat2cell(Vtnew,rnks,nout);  % insert into cell array for Vt

    % ===========================
    % Compute each weight matrix from its low-rank components
    wwfilts = cellfun(@mtimes,wU,wVt,'UniformOutput',false); % full weights in cell array
    ww = cell2mat(wwfilts);
    
    % Evaluate loss function & size of change
    fvalnew = 0.5*trace(ww'*XX*ww) - sum(sum((ww.*XY)));
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(opts.Display, 'iter')
        fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

