% demo4_testTrilinearRegress.m
%
% Tests out bilinear ridge regression with bilinearly parametrized coefficient vector; 

addpath tools
addpath fitting

% ---------------------------------------------
% Set dimensions & rank
n1 = 29; % number of time coefficients
n2 = 7; % number of spatial dim 1 coefficients
n3 = 21; % number of spatial dim 2 coefficients
nwtot = n1*n2*n3; % total number of filter coefficients
rnk = 1; % rank

% Make true weights (random low-rank matrix)
wtrue1 = gsmooth(randn(n1,rnk),2); % temporal filters
wtrue2 = gsmooth(randn(n2,rnk),2); % spatial filters, dim 1 
wtrue3 = gsmooth(randn(n3,rnk),2); % spatial filters, dim 2
wtensr = mkrank1tensor(wtrue1,wtrue2,wtrue3); % filter as a rank-1 3rd order tensor
wvec = vec(wtensr); % vectorized filter

%% test Kronecker formulas 
wvec1 = kron(vec(wtrue2*wtrue3'),speye(n1))*wtrue1;
wvec2 = kron(wtrue3,kron(speye(n2),wtrue1))*wtrue2;
wvec3 = kron(speye(n3),vec(wtrue1*wtrue2'))*wtrue3;

% Plot reconstructions to check they match
plot(1:nwtot,[wvec-wvec1,wvec-wvec2,wvec-wvec3]);
errs = [norm(wvec-wvec1), norm(wvec-wvec2), norm(wvec-wvec3)]; % should be zero

%% ---------------------------------------------
% Generate training data
nstim = 5000; % number of stimuli
signse = .5;  % stdev of observation noise
X = randn(nstim,nwtot); % stimuli
Y = X*wvec + randn(nstim,1)*signse; % observations

% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;
lambda = 1; % ridge parameter 

% ---------------------------------------------
% Estimate W: coordinate ascent
tic;
[what,w1hat,w2hat,w3hat] = trilinearRegress_coordAscent(XX,XY,[n1,n2,n3],rnk,lambda);
t1 = toc;

% ---------------------------------------------
% Plot filters and computer errors
% ---------------------------------------------

% Report timings
fprintf('\nTrilinear ridge regression test:\n');
fprintf('--------------------------------\n');
fprintf('computation time (coordinate ascent): %f\n', t1);

% % Plot true and estimated filters
subplot(411);
tt = 1:n1*n2*n3;
plot(tt, wvec,tt, what(:));
legend('true weights', 'estimate');
title('vectorized weights');
ylabel('coefficient');
ylabel('coefficient #');
box off;

% Plot filters as low-rank matrices
subplot(423); imagesc(wtrue1*wtrue2'); title('true w1 x w2');
subplot(424); imagesc(w1hat*w2hat'); title('estim w1 x w2');
subplot(425); imagesc(wtrue1*wtrue3'); title('true w1 x w3');
subplot(426); imagesc(w1hat*w3hat'); title('estim w1 x w2');
subplot(427); imagesc(wtrue2*wtrue3'); title('true w1 x w2');
subplot(428); imagesc(w2hat*w3hat'); title('estim w1 x w2');

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x(:)-y(:)).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(what(:),wvec));

