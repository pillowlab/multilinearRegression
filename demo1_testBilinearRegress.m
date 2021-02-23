% demo1_testBilinearRegress.m
%
% Tests out bilinear ridge regression with bilinearly parametrized coefficient vector; 

addpath tools
addpath fitting

nt = 50; % number of time coefficients
nx = 10; % number of spatial coefficients
rnk = 2; % rank

% Make true weights (random low-rank matrix)
A = gsmooth(randn(nt,rnk),2);  
B = gsmooth(randn(nx,rnk),2)';
nw = nt*nx;
wmat = A*B;
wtrue = vec(wmat);

subplot(211);
imagesc(wmat);

% Generate training data
nstim = 5000;
signse = 10;
X = randn(nstim,nw);
Y = X*wtrue + randn(nstim,1)*signse;
subplot(212);
plot(X*wtrue, Y, 'o');

% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;
lambda = 1;

% Estimate W: coordinate ascent
tic;
[what1,wt,wx] = bilinearRegress_coordAscent(XX,XY,[nt,nx],rnk,lambda);
t1 = toc;

% Estimate W: gradient-based ascent
opts = optimoptions(@fminunc, 'display', 'iter');  % set options (optional)
tic;
[what2,wt2,wx2] = bilinearRegress_grad(XX,XY,[nt,nx],rnk,lambda,opts);
t2 = toc;

% Report timings
fprintf('\ncomputation time (coordinate ascent): %f\n', t1);
fprintf('computation time (joint ascent):      %f\n', t2);

%% Plot filters and computer errors

subplot(211);
tt = 1:nt*nx;
plot(tt, wtrue,tt, what1(:),tt, what2(:),'--', 'linewidth', 2);
legend('true weights', 'estimate 1', 'estimate 2');
title('vectorized weights');
ylabel('coefficient');
ylabel('coefficient #');
box off;


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

