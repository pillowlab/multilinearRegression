% demo_multiLinearBilinearRegress.m
%
% Least squares regression with multiple bilinearly parametrized filters 

% Notes: 
% - Purely linear filters can be incorporated by setting either
%   nx or nt to 1 and rank = 1 for one of the filters
% - Additive constant can be obtained by adding a filter with nt=nx=1 and
%   rank=1

setpath; % set path 

% Set up true filter sizes and ranks
nt = [20,8,20];  % temporal lengths
nx = [10,15,20];  % spatial lengths
rnk = [2,1,3];  % rank of each filter

nws = nt.*nx;  % number of total filter elements for each filter
nfilts = length(nws);  % number of total filters
nwtot = sum(nws); % total number of filter coefficients

% -------------------------------------
% Make true filters
flts = struct; % structure for filters
wtrue = []; % true filter vector
for jj = 1:nfilts
    flts(jj).wx = gsmooth(randn(nt(jj),rnk(jj)),2); % temporal filters
    flts(jj).wt = gsmooth(randn(nx(jj),rnk(jj)),2)'; % spatial filters
    flts(jj).wfilt = flts(jj).wx*flts(jj).wt; % filter
    
    % assemble full filter vector
    wtrue = [wtrue; flts(jj).wfilt(:)];
end

% -------------------------------------
% Generate training data
nstim = 5000; % number of stimuli
signse = 5;  % stdev of observation noise
X = randn(nstim,nwtot); % stimuli
Y = X*wtrue + randn(nstim,1)*signse; % observations

% Compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% -------------------------------------
% Estimate by Coordinate Ascent 
lambda = 10;  % ridge parameter
tic;
[wtfit,wxfit,wvecfit] = bilinearMultifiltRegress_coordAscent(XX,XY,nt,nx,rnk,lambda);
toc;

% -------------------------------------
% Make plots & compute error

% Plot true weights and estimate as a single (long) vector.
subplot(311); 
plot(1:nwtot,wtrue,1:nwtot,wvecfit,'--','linewidth',2);
legend('true weights', 'estimate'); 
set(gca,'xlim', [0 nwtot]);
title('vectorized weights and estimate');

% plot true weights as images
subplot(334);
imagesc(flts(1).wfilt);axis image;
title('true filter 1'); box off;
ylabel('temporal coeff'); 
subplot(335);
imagesc(flts(2).wfilt); axis image;
title('true filter 2'); box off;
subplot(336);
imagesc(flts(3).wfilt); axis image;
title('true filter 3'); box off;

% plot estimated weights as images
subplot(337);
imagesc(wtfit{1}*wxfit{1});axis image;
title('estim filter 1'); box off;
ylabel('temporal coeff'); 
xlabel('space coeff');
subplot(338);
imagesc(wtfit{2}*wxfit{2}); axis image;
title('estim filter 2'); box off;
xlabel('space coeff');
subplot(339);
imagesc(wtfit{3}*wxfit{3}); axis image;
title('estim filter 3'); box off;
xlabel('space coeff');
% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(wvecfit, wtrue));
