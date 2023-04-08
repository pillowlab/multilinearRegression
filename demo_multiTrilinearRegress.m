% demo_multiTrilinearRegress.m
%
% Tests out regression with multiple rank-1 trilinearly parametrized filters
% (Note: only handles rank 1 at present)

% Notes: 
%
%  - regular linear filters can be incorporated by setting n2 = n3 = 1.
%  - additive constant can be handled by a filter with n1 = n2 = n3 = 1.

setpath; % set path 

% Set up sizes and ranks for 4 filters
n1 = [13,11,11,1];  % # of coefficients, 1st tensor dim
n2 = [7,7,7,1];  % # of coefficients, 2nd tensor dim
n3 = [7,7,7,1];  % # of coefficients, 3rd tensor dim
rnk = [1,1,1,1]; % ranks

ncoeffsperfilt = n1.*n2.*n3;  % number of total filter elements for each filter
nfilts = length(ncoeffsperfilt);  % number of total filters
ncoeffstot = sum(ncoeffsperfilt); % total number of filter coefficients

% -------------------------------------
% Make true filters
flts = struct; % structure for filters
wtrue = []; % true filter vector
for jj = 1:nfilts
    flts(jj).w1 = gsmooth(randn(n1(jj),rnk(jj)),2); % temporal filters
    flts(jj).w2 = gsmooth(randn(n2(jj),rnk(jj)),2); % spatial filters
    flts(jj).w3 = gsmooth(randn(n3(jj),rnk(jj)),2); % spatial filters
    flts(jj).wfilt = mkrank1tensor(flts(jj).w1,flts(jj).w2,flts(jj).w3); % filter
    
    % assemble full filter vector by stacking
    wtrue = [wtrue; flts(jj).wfilt(:)];
end

%% -------------------------------------
% Generate training data
nstim = 5000; % number of stimuli
signse = 2;  % stdev of observation noise
X = randn(nstim,ncoeffstot); % stimuli
Y = X*wtrue + randn(nstim,1)*signse; % observations

% Compute sufficient statistics
XX = X'*X;
XY = X'*Y;

%% -------------------------------------
% Estimate by Coordinate Ascent 
lambda = 1;  % ridge parameter
tic;
[w1fit,w2fit,w3fit,wvecfit] = trilinearMultifiltRegress_coordAscent(XX,XY,n1,n2,n3,rnk,lambda);
toc;

% -------------------------------------
% Make plots & compute error

% Plot true weights and estimate as a single (long) vector.
subplot(411); 
plot(1:ncoeffstot,wtrue,1:ncoeffstot,wvecfit,'--','linewidth',2);
legend('true weights', 'estimate'); 
set(gca,'xlim', [0 ncoeffstot]);
title('vectorized weights and estimate');

% % plot 2D faces as images
f1_12 = flts(1).w1*flts(1).w2'; f1_13 = flts(1).w1*flts(1).w3'; 
f2_12 = flts(2).w1*flts(2).w2'; f2_13 = flts(2).w1*flts(2).w3'; 
f3_12 = flts(3).w1*flts(3).w2'; f3_13 = flts(3).w1*flts(3).w3'; 
subplot(445); imagesc(f1_12); title('true filt 1: w1 x w2');
subplot(447); imagesc(f1_13); title('true filt 1: w1 x w3');
subplot(449); imagesc(f2_12); title('true filt 2: w1 x w2');
subplot(4,4,11); imagesc(f2_13); title('true filt 2: w1 x w3');
subplot(4,4,13); imagesc(f3_12); title('true filt 3: w1 x w2');
subplot(4,4,15); imagesc(f3_13); title('true filt 3: w1 x w3');

% plot fitted 2D faces as images (with sign flips if necessary)
k1_12 = w1fit{1}*w2fit{1}'; k1_13 = w1fit{1}*w3fit{1}'; 
k2_12 = w1fit{2}*w2fit{2}'; k2_13 = w1fit{2}*w3fit{2}'; 
k3_12 = w1fit{3}*w2fit{3}'; k3_13 = w1fit{3}*w3fit{3}'; 
subplot(446); imagesc(k1_12 * sign(k1_12(:)'*f1_12(:))); title('fit 1: w1 x w2');
subplot(448); imagesc(k1_13 * sign(k1_13(:)'*f1_13(:))); title('fit 1: w1 x w3');
subplot(4,4,10); imagesc(k2_12 * sign(k2_12(:)'*f2_12(:))); title('fit 2: w1 x w2');
subplot(4,4,12); imagesc(k2_13 * sign(k2_13(:)'*f2_13(:))); title('fit 2: w1 x w3');
subplot(4,4,14); imagesc(k3_12 * sign(k3_12(:)'*f3_12(:))); title('fit 3: w1 x w2');
subplot(4,4,16); imagesc(k3_13 * sign(k3_13(:)'*f3_13(:))); title('fit 3: w1 x w3');

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(wvecfit, wtrue));
