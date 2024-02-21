% demo_LowRankAR1.m
%
% Low-rank vector autoregressive model, defined by:
%
%  x_t = W x_{t-1} + noise, 
% 
%  where W is low rank.
%
% Fitting this model is a special case of reduced rank regression (RRR),
% where the number of input neurons is the same as the number of output
% neurons.  
%
% The purpose of this script is to illustrate how to use different
% optimization methods to fit this model. 


%%  Setup 

setpath; % set path 

% Set dimensions & rank
nx = 50; % number of neurons
rnk = 3;  % rank of weights

% Make true weights (random low-rank matrix)
wx = gsmooth(randn(nx,rnk),3); % column vectors
wy = gsmooth(randn(nx,rnk),3)'; % row vectors

wwtruemat = wx*wy; % true autoregressive weight matrix

% scale down so abs(eigenvalues) are <= 1
[u,s] = eig(wwtruemat,'vector'); % get eigenvectors and eigenvals
s = s/max(abs(s))*.99; % set largest eigenvalue to lie inside unit circle (enforcing stability)
s(real(s)<0) = -s(real(s)<0); % set real parts to be positive (encouraging smoothness)
wwtruemat = real(u*(diag(s)/u));  % reconstruct ww from eigs and eigenvectors


%% Generate training data

ntbins = 1e3; % number of time bins
signse = 2;  % stdev of observation noise
rr = zeros(ntbins+1,nx); % neural data
rr(1,:) = randn(1,nx)*signse;

for jj = 2:ntbins+1
    rr(jj,:) = rr(jj-1,:)*wwtruemat + randn(1,nx)*signse; %
end

% Create inputs and outputs for regression problem
X = rr(1:ntbins,:); % inputs
Y = rr(2:ntbins+1,:);  % outputs

% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;

%% Run RRR

% compute LS solution
wls = XX\XY;

[~,~,vrrr] = svd(Y'*X*wls);  % perform SVD of relevant matrix
vrrr = vrrr(:,1:rnk);      % get column vectors
urrr= (X'*X)\(X'*Y*vrrr);  % get row vectors
wrrr = urrr*vrrr';  % construct full RRR estimate

%% Estimate W using alternating coordinate ascent (bilinear optimization)

% Make the necessary design matrix
XXvec = kron(speye(nx),XX); % design matrix for vectorized problem
XYvec = XY(:);  % vectorized XY matrix

% set options
opts.MaxIter = 50;
opts.TolFun = 1e-8;
opts.Display = 'off';

lambda = 0;  % set the ridge parameter (zero = 'no regularization')

% Find estimate by alternating coordinate ascent
[wbilin,ubilin,vbilin] = bilinearRegress_coordAscent_fast(XXvec,XYvec,[nx,nx],rnk,lambda,opts);  


%% Make plots and compute R^2 

subplot(221); imagesc(wwtruemat); 
title(sprintf('true filter (rank=%d)',rnk));

subplot(222); imagesc(wls); 
title('least squares');

subplot(223); imagesc(wrrr); 
title('RRR');

subplot(224); imagesc(wbilin);
title('bilinear optim');

% Compute R^2 between true and estimated weights
wwtruevec = vec(wwtruemat); % vectorized filter
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x)(1-msefun(x(:),wwtruevec)./msefun(wwtruevec,mean(wwtruevec)));

fprintf('\nPerformance comparison (R^2):\n');
fprintf('----------------------------\n');
fprintf(' least-squares: %.3f\n',r2fun(wls));
fprintf('           RRR: %.3f\n',r2fun(wrrr));
fprintf('bilinear optim: %.3f\n',r2fun(wbilin));

% Note: RRR and bilinear optim estimates should match!