% demo_RRR.m
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
ww = struct;  % structure for true filters
ww.nin = [20,30];  % number of neurons in each input population
ww.nout = [100];   % number of neurons in output population
ww.rnk = [1,2];  % rank of each filter

ww.ninpops = length(ww.nin); % number of input populations
ww.nintot = sum(ww.nin);     % total number of input neurons

% =======  Make true filters ======
ww.fullflt = zeros(nintot,ww.nout);
nincum = 0; % cumulative # of input neurons
for jj = 1:ww.ninpops
    ww.flts(jj).u = gsmooth(randn(ww.nin(jj),ww.rnk(jj)),2); % v filter
    ww.flts(jj).v = gsmooth(randn(ww.nout,ww.rnk(jj)),2); % u filter
    ww.flts(jj).flt = ww.flts(jj).u*ww.flts(jj).v';  % filter for population jj
    
    % add this filter to the full all-population filter (by stacking them)
    ww.fullflt(nincum+1:nincum+ww.nin(jj),:) = ww.flts(jj).flt;
    nincum = nincum+ww.nin(jj); % update cumulative # of input neurons
end

%% ======= Generate training data by simulating from the model =========
nstim = 500; % number of trials 
signse = 3;  % stdev of observation noise
X = randn(nstim,nintot); % input neurons

% Compute model output
Y = X*ww.fullflt + randn(nstim,ny)*signse; % output neurons (observations)


% Pre-compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% compute LS solution
wls = XX\XY;




