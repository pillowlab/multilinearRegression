function [wU,wVt,wwfilts] = bilinearMultifiltRRR_coordAscent(Xin,Yout,rnks,lambda,opts)
% [wU,wVt,wwfilts] = bilinearMultifiltRRR_coordAscent(Xin,Yout,nin,rnks,lambda,opts)
% 
% Computes low-rank regression estimate using coordinate ascent in the
% multi-filter reduced rank regression (RRR) problem setting 
% the parameter vector.
%
% Finds solution to: 
%   argmin_{W_i} ||Y - \sum_i Xi Wi||_F^2 + lambda*\sum_i ||W_i||_F^2
% where the matrices {W_i} are all low rank, parametrized as
%   W_i = U_i V_i^T
%
% Inputs:
% -------
%   X - input population design matrices (T x nintot)
%   Y - output population response matrix (T x nout) 
%   nin - # of input neurons in each input populatin
%   rnks - rank of each low-rank filter
%
%   lambda - ridge parameter (optional)  (NOT YET SUPPORTED)
%
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:
% -------
%   wU  = column vector matrices (cell array)
%   wVt = row vector matrices    (cell array)
%   wwfilts = low rank filters   (cell array)


% ---------------------------------------------------
% set optimization options
% ---------------------------------------------------
if (nargin < 5) || isempty(opts)
    opts.default = true;
end
if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

% ---------------------------------------------------
% Extract sizes of inputs
% ---------------------------------------------------

nin = cellfun(@(x)size(x,2),Xin); % get # of cells in each population
ninpops = length(nin); % number of input populations
nintot = sum(nin);     % total number of input neurons
nout = size(Yout,2);

% check inputs
if length(rnks)~=ninpops
    error('length of ``rnks'' doesn''t match # of input populations (# cells in Xin)');
end

% ---------------------------------------------------
% Initialize using SVD of ridge regression estimate
% ---------------------------------------------------
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

% ---------------------------------------------------
% Set up coordinate ascent 
% ---------------------------------------------------

% Convert initial filters into a matrix
ww0 = cellfun(@mtimes,wU,wVt,'UniformOutput',false);
ww0 = cell2mat(ww0);

% Evaluate the loss function at the start of training
fval = 0.5*trace(ww0'*XX*ww0) - sum(sum((ww0.*XY)));
fchange = inf; % change in function value
iter = 1;  % iteration counter

% Report initial loss (if 'Display' set to 'iter')
if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of multi-filter RRR loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

% % define some useful cell functions and a cell array we'll need
fun_Ureshape = @(v,r)(reshape(v,[],r));    % function to reshape U matrices 
fun_mtimesTrp = @(A,B)(vec(mtimes(A,B'))); % function to compute kronecker of transpose with matrix
rnks_cell = num2cell(rnks(:));          % filter ranks as cell array
XXc = mat2cell(XX,nin,nin);
XYc = mat2cell(XY,nin,nout);

% ---------------------------------------------------
% Optimize:  Run alternating coordinate ascent on U and Vt
% ---------------------------------------------------

while (iter <= opts.MaxIter) && (fchange > opts.TolFun)
    
    % ===========================
    % Update U (column vectors)
    % ===========================
        
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
    % ===========================
    
    % project Xin onto corresponding U matrices and do regression
    Mu = cell2mat(cellfun(@mtimes,Xin,wU','UniformOutput',false)); % design matrix for estimating Vt
    Vtnew = (Mu'*Mu)\(Mu'*Yout); % solve regression problem for Vt
    wVt = mat2cell(Vtnew,rnks,nout);  % insert into cell array for Vt

    % ===========================
    % Compute each weight matrix from its low-rank components
    % ===========================
    
    wwfilts = cellfun(@mtimes,wU,wVt,'UniformOutput',false); % full weights in cell array
    
    % ===========================
    % Evaluate loss function & size of change
    % ===========================
    
    ww = cell2mat(wwfilts);  % put filters into a single weight matrix
    fvalnew = 0.5*trace(ww'*XX*ww) - sum(sum((ww.*XY)));
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    
    % Report change in error (if desired)
    if strcmp(opts.Display, 'iter')
        fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

