% demo_RRRmultifilter.m
%
% Examine multi-filter Reduced Rank Regression (RRR), defined by the
% low-rank multivariate regression problem:
%  
%  y = W_1 x_1 + W_2 x_2 + ... + W_k x_k + noise
%
% y is an n-vector, x_1, ..., x_k are vectors of lenth m1, ..., mk,
% and W_1 \in {n x m1}, ..., W_k \in {n x mk} are
% regression weight matrices of rank r1, ..., rk.
%
% When k=1, this is standard RRR.


%% Setup.

setpath; % set path 

% ==== Set dimensions & rank =========

% Set up true filter sizes and ranks
nin = [80,90];  % number of neurons in each input population
nout = 100;      % number of neurons in the output population
rnks = [2,3];  % rank of each filter

ninpops = length(nin); % number of input populations
nintot = sum(nin);     % total number of input neurons

% =======  Make true filters ======
wwlowrank = struct; % struct for storing filters
wwtrue = cell(ninpops,1); % cell array for filters
for jj = 1:ninpops
    wwlowrank(jj).u = gsmooth(randn(nin(jj),rnks(jj)),2); % u filter
    wwlowrank(jj).v = gsmooth(randn(nout,rnks(jj)),2); % v filter
    wwtrue{jj} = wwlowrank(jj).u*wwlowrank(jj).v';  % filter for population jj
end
wwtruemat = cell2mat(wwtrue); % concatenate filters into single matrix

%% ======= Generate training data by simulating from the model =========

nstim = 2000; % number of trials 
signse = 5;  % stdev of observation noise

% Generate input population responses
Xin = cell(1,ninpops);
for jj = 1:ninpops
    Xin{jj} = randn(nstim,nin(jj)); % input neurons
end
Xfull = cell2mat(Xin);

% Compute simulated model output
Yout = Xfull*wwtruemat;

% % --------------------------
% % Equivalent code, applying each filter to each population
% Yout2 = zeros(nstim,nout);
% for jj = 1:ninpops
%     Yout2 = Yout2 + Xin{jj}*fltrs{jj};
% end
% plot(Yout(:)-Yout2(:)); % should be all zeros
% % --------------------------

% Add observation noise
Yout = Yout + randn(nstim,nout)*signse; 

%% Estimate full filter using generic ridge regression

lambda = 10;  % set ridge parameter

% Compute sufficient statistics
XX = Xfull'*Xfull;  % stimulus covariance
XY = Xfull'*Yout;   % stim-response cross-covariance

% compute ridge solution
wridge = (XX+lambda*eye(nintot))\XY; % ridge regression solution
wridgemat = reshape(wridge,nintot,nout); % reshape as a single matrix

%% Estimate filters using coordinate ascent for bilinear regression model

% Here we need to make a single design matrix X such that we can rewrite
% the optimization as
%                        min ||vec(Y) - X vec(W)||^2
% where vec(W) represents the individual filters vectorized and then
% stacked on top of each other. To do this, we need to form X out of a
% concatenation of kronecker matrices, one for each input population:
%             X = [kron(I,X1) kron(I,X2) ...  kron(I,Xk)]

tic; % time building the design matrix
Xkron = cell(1,ninpops); % cell array for per-population design matrices
for jj = 1:ninpops
    Xkron{jj} = kron(speye(nout),Xin{jj}); % design matrix for jj'th input population
end
Xkronfull = cell2mat(Xkron);  % concatenate design matrices into 1 matrix
XXkron = Xkronfull'*Xkronfull;  % 2nd moment matrix
XYkron = Xkronfull'*vec(Yout);  % cross-covariance matrix
t1 = toc;
fprintf('Time to build kronecker design matrix: %.4f sec\n\n',t1);

% Perform optimization of the weights
tic; % time optimization
[wtfit,wxfit,wvecfit] = bilinearMultifiltRegress_coordAscent(XXkron,XYkron,nin,nout*ones(1,ninpops),rnks,lambda);
t2 = toc;
fprintf('Time for bilinear optimization: %.4f sec\n\n',t2);


%% Make plots (assuming 2 filters in model)

% vectorize true filters and concatenate
wwvectrue = cell2mat(cellfun(@vec, wwtrue, 'UniformOutput', false));
nwtot = length(wwvectrue);

% vectorize ridge filter estimates and concatenate 
wridgefilts = mat2cell(wridgemat,nin,nout);  % convert ridge matrix into cell array for individual filters
wridgevec = cell2mat(cellfun(@vec, wridgefilts, 'UniformOutput', false)); % vectorize   

% Plot true weights and estimate as a single (long) vector.
subplot(311); 
plot(1:nwtot,wwvectrue,1:nwtot,wridgevec,'--',1:nwtot,wvecfit,'--','linewidth',2);
legend('true weights', 'ridge', 'bilin');
set(gca,'xlim', [0 nwtot]);
title('vectorized weights and estimate');

% plot true weights as images
subplot(334);
imagesc(wwtrue{1});axis image;
title(sprintf('true pop 1 filter (rank=%d)',rnks(1))); box off;
ylabel('input neuron #'); xlabel('output neuron #'); 
subplot(337); % filter 2
imagesc(wwtrue{2}); axis image;
title(sprintf('true pop 2 filter (rank=%d)',rnks(2))); box off;
ylabel('input neuron #'); xlabel('output neuron #'); 

% plot ridge estimates
subplot(335);
imagesc(wridgemat(1:nin(1),:)); axis image;
title('ridge filt 1'); box off;
ylabel('input neuron #'); xlabel('output neuron #'); 
subplot(338); % filter 2
imagesc(wridgemat(nin(1)+1:end,:)); axis image;
title('ridge filt 2'); box off;
ylabel('input neuron #'); xlabel('output neuron #'); 

% plot reduced-rank estimates
subplot(336);
imagesc(wtfit{1}*wxfit{1});axis image;
title('low-rank estim filter 1'); box off;
ylabel('input neuron #'); 
xlabel('output neuron #'); 
subplot(339);
imagesc(wtfit{2}*wxfit{2}); axis image;
title('low-rank estim filter 2'); box off;
ylabel('input neuron #'); xlabel('output neuron #'); 

% Compute R^2 between true and estimated weights
msefun = @(x,y)(mean((x-y).^2));
r2fun = @(x,y)(1-msefun(x,y)./msefun(y,mean(y)));
fprintf('R-squared (ridge):    %.3f\n',r2fun(wridgevec, wwvectrue));
fprintf('R-squared (bilinear): %.3f\n',r2fun(wvecfit, wwvectrue));
