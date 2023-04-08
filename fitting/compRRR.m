function [ww,urrr,vrrr] = compRRR(X,Y,rnk)
% [ww,urrr,vrrr] = compRRR(X,Y,rnk)
% 
% Compute reduced rank regression weights
%
% Finds solution to argmin_w ||Y - X*w||^2 
% where w = urrr*vrrr' for some low-rank r
%
% Inputs:
% -------
%   X [nt x nx ] - inputs or regressors
%   Y [nt x ny ] - outputs or predictors
%   rnk [1 x 1 ] - desired rank of weight matrix
%
% Outputs:
% -------
%   ww = weights 
%   urrr = column vectors
%   vrrr = row vectors

% Compute sufficient statistics
XX = X'*X;
XY = X'*Y;

% compute least-squares weights
wls = XX\XY;

% Compute RRR weights
[~,~,vrrr] = svd(Y'*X*wls);  % perform SVD of relevant matrix
vrrr = vrrr(:,1:rnk);      % column vectors
urrr= (X'*X)\(X'*Y*vrrr);  % row vectors
ww = urrr*vrrr';  % construct full RRR estimate
