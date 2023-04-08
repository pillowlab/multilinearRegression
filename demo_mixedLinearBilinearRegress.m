% demo_mixedLinearBilinearRegress.m
%
% Tests out bilinear ridge regression where part of coefficient vector is
% partly parametrized bilinearly and partly parametrized linearly

setpath;  % set path

% Set up true weights: part of the filter is bilinear, part is linear
nt = 50;  % temporal length
nx = 20;  % spatial length
rnk = 2;  % rank
nwlin = 100;  % length of linear part

nwbi = nt*nx;  % number of filter elements in bilinear part
nwtot = nwbi+nwlin; % total number of filter elements

iibi = floor(nwlin/2)+(1:nwbi);  % indices for bilinear elements
iilin = setdiff(1:nwtot,iibi);   % indices for linear elements

% ----------------------------------------------
% Make true filters

wt = gsmooth(randn(nt,rnk),2); % temporal filters
wx = gsmooth(randn(nx,rnk),2)'; % spatial filters
wlin = gsmooth(randn(nwlin,1),10); % purely linear component

wbi = wt*wx;  % bilinear filter
wtrue = zeros(nwtot,1);  % composite filter
wtrue(iibi) = vec(wbi); % the regression coefficients
wtrue(iilin) = wlin;

% ----------------------------------------------
% Generate training data

nstim = 5000; % number of stimuli
signse = 10;  % stdev of observation noise
X = randn(nstim,nwtot); % stimuli
Y = X*wtrue + randn(nstim,1)*signse; % observations

% Compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% ----------------------------------------------
% Estimate by Coordinate Ascent 
lambda = 10;  % ridge parameter
tic;
[what1,wt1,wx1,wlin1] = bilinearMixRegress_coordAscent(XX,XY,[nt,nx],rnk,iibi,lambda);
t1 = toc;

% ----------------------------------------------
% Estimate by Gradient Ascent
tic;
[what2,wt2,wx2,wlin2] = bilinearMixRegress_grad(XX,XY,[nt,nx],rnk,iibi,lambda);
t2 = toc;


% ----------------------------------------------
% Make plots & Report Error 
% ----------------------------------------------

% Report timings
fprintf('\nMixed bilinear and linear regression:\n');
fprintf('-------------------------------------\n');
fprintf('computation time (coordinate ascent): %f\n', t1);
fprintf('computation time (joint ascent):      %f\n', t2);

% Make plot: linear filters
subplot(221);
plot(1:nwlin,wlin,1:nwlin,wlin1,1:nwlin,wlin2, '--', 'linewidth', 2);
legend('true','estim 1 (coord asc.)', 'estim 2 (joint asc.)'); box off;
xlabel('coefficient #');
title('linear weights');

% Make plots: bilinearly parametrized filters
subplot(234); imagesc(wbi);
title('true bilinear wts'); box off;
ylabel('time (bins)'); xlabel('space (bins)');
subplot(235); imagesc(wt1*wx1); box off;
title('bilin estim (coord. asc.)');
subplot(236); imagesc(wt2*wx2); box off;
title('bilin estim (joint asc.)');

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(what1(:), wtrue));
fprintf('R-squared (joint ascent):      %.3f\n',r2fun(what2(:), wtrue));
