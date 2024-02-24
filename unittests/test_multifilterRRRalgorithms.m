% test_multifilterRRRalgorithms.m
%
% Unit tests for multi-filter RRR algorithms. 
%
% Goal of these algorithms is to solve the low-rank multivariate regression problem:
%  
%  ||y - W1 x1 + W2 x2 + ... + Wk xk|| + noise
%
% y is an n-vector, x1, ..., xk are vectors of lenth m1, ..., mk,
% and W1 \in {n x m1}, ..., Wk \in {n x mk} are
% regression weight matrices of rank r1, ..., rk.
%
% When k=1, this is standard RRR.

addpath ../fitting
addpath ../tools
addpath ..

TOL = 1e-6;  % tolerance for unit test

% ==== Set dimensions & rank =========

% Set up true filter sizes and ranks
nin = [15,25];  % number of neurons in each input population
nout = 40;     % number of neurons in the output population
rnks = [3,2];  % rank of each filter
nstim = 1000; % number of trials 
signse = 2;  % stdev of observation noise
lambda = 10;  % set ridge parameter

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


%% Estimate filters generic algorithm for bilinear regression model (slow)

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
fprintf('Time to build kronecker design matrix: %.4f sec\n',t1);

% Perform optimization of the weights
opts.Display = 'off';
tic; % time optimization
[wUfit1,wVtfit1,wwfitvec1,fval1] = bilinearMultifiltRegress_coordAscent(XXkron,XYkron,nin,nout*ones(1,ninpops),rnks,lambda,opts);
t2 = toc;
fprintf('Time for standard bilinear optimization algorithm: %.4f sec\n',t2);

%% Estimate filters using algorithm specific to RRR with multiple filters (fast)

% This version uses an algorithm which explicitly takes into account that
% this is multivariate regression, so Yout is a matrix whose columns would
% (were the problem not low rank) correspond to independent regression
% problems.

tic;
[wUfit2,wVtfit2,wwfilts2,fval2] = bilinearMultifiltRRR_coordAscent(Xin,Yout,rnks,lambda,opts);
t3 = toc;
fprintf('Time for multi-filter RRR optimization: %.4f sec\n',t3);

%% Report outcome of unit test 

% vectorize RRR estimated filters
wwfitvec2 = cell2mat(cellfun(@vec, wwfilts2, 'UniformOutput', false));

maxFiltError = max(abs(wwfitvec1-wwfitvec2));
fvalError = abs(fval1-fval2);

if (maxFiltError<TOL) && (fvalError<TOL)
    fprintf('test_multifilterRRRalgorithms.m test PASSED\n'); 
else
    warning('test_multifilterRRRalgorithms.m test FAILED');
    fprinttf('Filter error: %.5f\n',  maxFiltError);
    fprinttf('  Fval error: %.5f\n',  fvalError);
end
