% demo_bilinearRegress.m
%
% Tests out bilinear ridge regression with bilinearly parametrized coefficient vector

setpath; % set path 

% ---------------------------------------------
% Set dimensions & rank
nt = 100; % number of time coefficients
nx = 20; % number of spatial coefficients
rnk = 2; % rank

% Make true weights (random low-rank matrix)
wt = gsmooth(randn(nt,rnk),2); % temporal filters
wx = gsmooth(randn(nx,rnk),2)'; % spatial filters
nwtot = nt*nx; % total number of filter coefficients
wmat = wt*wx; % filter as a matrix
wtrue = vec(wmat); % vectorized filter

% ---------------------------------------------
% Generate training data
nstim = 5000; % number of stimuli
signse = 10;  % stdev of observation noise
X = randn(nstim,nwtot); % stimuli
Y = X*wtrue + randn(nstim,1)*signse; % observations

% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;
lambda = 1; % ridge parameter 

% ---------------------------------------------
% Estimate W: coordinate ascent
tic;
[what1,wt1,wx1] = bilinearRegress_coordAscent_fast(XX,XY,[nt,nx],rnk,lambda);
t1 = toc;

% % slower method
% tic;
% [what1b,wt1b,wx1b] = bilinearRegress_coordAscent(XX,XY,[nt,nx],rnk,lambda);
% t1b = toc;

% ---------------------------------------------
% Estimate W: gradient-based ascent
opts = optimoptions(@fminunc, 'display', 'iter');  % set options (optional)
tic;
[what2,wt2,wx2] = bilinearRegress_grad(XX,XY,[nt,nx],rnk,lambda,opts);
t2 = toc;

% ---------------------------------------------
% Plot filters and computer errors
% ---------------------------------------------

% Report timings
fprintf('\nBilinear ridge regression test:\n');
fprintf('--------------------------------\n');
fprintf('computation time (coordinate ascent): %f\n', t1);
fprintf('computation time (joint ascent):      %f\n', t2);

% Plot vectorized true and estimated filters
subplot(211);
tt = 1:nt*nx;
plot(tt, wtrue,tt, what1(:),tt, what2(:),'--', 'linewidth', 2);
legend('true weights', 'estimate 1', 'estimate 2');
title('vectorized weights');
ylabel('coefficient');
ylabel('coefficient #');
box off;

% Plot filters as low-rank matrices
subplot(234); imagesc(wmat);
title('true weights'); box off;
ylabel('time (bins)'); xlabel('space (bins)');
subplot(235); imagesc(what1); box off;
title('estimate 1 (coord. asc.)');
subplot(236); imagesc(what2); box off;
title('estimate 2 (joint asc.)');

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(what1(:), wtrue));
fprintf('R-squared (joint ascent):      %.3f\n',r2fun(what2(:), wtrue));

