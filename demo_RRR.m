% demo_RRR.m
%
% Examine Reduced Rank Regression, which is a special case of bilinear
% regression, and compare the closed-form RRR solution with bilinar
% optimization. 

setpath; % set path 

% ---------------------------------------------
% Set dimensions & rank

nx = 50; % number of input neurons
ny = 30; % number of output neurons
rnk = 3;  % rank

% ----------------------------------------------
% Make true weights (random low-rank matrix)
wx = gsmooth(randn(nx,rnk),3); % column vectors
wy = gsmooth(randn(ny,rnk),3)'; % row vectors

nwtot = nx*ny; % total number of filter coefficients
wtruemat = wx*wy; % filter as a matrix
wtruevec = vec(wtruemat); % vectorized filter

% ---------------------------------------------
% Generate training data
nstim = 500; % number of trials 
signse = 3;  % stdev of observation noise
X = randn(nstim,nx); % input neurons
Y = X*wtruemat + randn(nstim,ny)*signse; % output neurons (observations)

% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% compute LS solution
wls = XX\XY;

%% Run reduced rank regression

[wrrr,urrr,vrrr] = compRRR(X,Y,rnk);

%% Estimate W using bilinear optimization (coordinate ascent algorithm)

% Make the necessary design matrix
XXvec = kron(speye(ny),XX); % design matrix for vectorized problem
XYvec = XY(:);  % vectorized XY matrix

% set options
opts.MaxIter = 50;
opts.TolFun = 1e-8;
opts.Display = 'off';

lambda = 0;  % remove the ridge parameter
[wbilin,ubilin,vbilin] = bilinearRegress_coordAscent_fast(XXvec,XYvec,[nx,ny],rnk,lambda,opts);  % solve by bilinear optimization


%% Make plots and compute R^2 

subplot(221); imagesc(wtruemat); 
title(sprintf('true filter (rank=%d)',rnk));

subplot(222); imagesc(wls); 
title('least squares');

subplot(223); imagesc(wrrr); 
title('RRR');

subplot(224); imagesc(wbilin);
title('bilinear optim');

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x)(1-msefun(x(:),wtruevec)./msefun(wtruevec,mean(wtruevec)));

fprintf('\nPerformance comparison (R^2):\n');
fprintf('----------------------------\n');
fprintf(' least-squares: %.3f\n',r2fun(wls));
fprintf('           RRR: %.3f\n',r2fun(wrrr));
fprintf('bilinear optim: %.3f\n',r2fun(wbilin));

