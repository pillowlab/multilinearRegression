% demo2_testMixedLinearBilinearRegress.m
%
% Tests out bilinear ridge regression where part of coefficient vector is
% partly parametrized bilinearly and the rest is linear.  


% Set up true weights: part of the filter is bilinear, part is linear
nt = 50;  % temporal length
nx = 20;  % spatial length
rnk = 2;  % rank
nwlin = 100;  % length of linear part

nwbi = nt*nx;  % number of filter elements in bilinear part
nwtot = nwbi+nwlin; % total number of filter elements

iibi = floor(nwlin/2)+(1:nwbi);  % indices for bilinear elements
iilin = setdiff(1:nwtot,iibi);   % indices for linear elements

% Make filters
A = gsmooth(randn(nt,rnk),2); % temporal filters
B = gsmooth(randn(nx,rnk),2)'; % spatial filters
wlin = gsmooth(randn(nwlin,1),10); % purely linear component

wbi = A*B;  % bilnear filter
wtrue = zeros(nwtot,1);  % composite filter
wtrue(iibi) = vec(wbi); % the regression coefficients
wtrue(iilin) = wlin;

% Generate training data
nstim = 5000; % number of stimuli
signse = 10;  % stdev of observation noise
X = randn(nstim,nwtot); % stimuli
Y = X*wtrue + randn(nstim,1)*signse; % observations

% Compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% Estimate by Coordinate Ascent 
lambda = 10;  % ridge parameter
tic;
[what1,wt1,wx1,wlin1] = bilinearMixRegress_coordAscent(XX,XY,[nt,nx],rnk,iibi,lambda);
t1 = toc;

% Estimate by Gradient Ascent
tic;
[what2,wt2,wx2,wlin2] = bilinearMixRegress_grad(XX,XY,[nt,nx],rnk,iibi,lambda);
t2 = toc;

% Report timings
fprintf('\nMixed bilinear and linear regression:\n');
fprintf('-------------------------------------\n');
fprintf('computation time (coordinate ascent): %f\n', t1);
fprintf('computation time (joint ascent):      %f\n', t2);


% Assemble full weights for both cases

% Make plots
subplot(221);
plot(1:nwlin,wlin,1:nwlin,wlin1,1:nwlin,wlin2, '--', 'linewidth', 2);
legend('true','estim 1 (coord asc.)', 'estim 2 (joint asc.)'); box off;
xlabel('coefficient #');
title('linear weights');

subplot(234); imagesc(wbi);
title('true bilinear wts'); box off;
ylabel('time (bins)'); xlabel('space (bins)');
subplot(235); imagesc(wt1*wx1); box off;
title('bilin estim (coord. asc.)');
subplot(236); imagesc(wt2*wx2); box off;
title('bilin estim (joint asc.)');



%plot([wtrue what1 what2]);
%subplot(212);
%imagesc([wbi, wt*wx, wt2*wx2]);

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('\nR-squared (coordinate ascent): %.3f\n',r2fun(what1(:), wtrue));
fprintf('R-squared (joint ascent):      %.3f\n',r2fun(what2(:), wtrue));
