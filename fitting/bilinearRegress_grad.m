function [w_hat,wt,wx] = bilinearRegress_grad(xx,xy,wDims,rnk,lambda,opts)
% [w_hat,wt,wx] = bilinearRegress_grad(xx,xy,wDims,p,lambda,opts)
% 
% Computes regression estimate with a bilinear parametrization of the
% parameter vector.  Uses grad and Hessian information to
% perform ascent.
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where w is parametrized as vec( wt*wx')
%
% Inputs:
%   xx - autocorrelation of design matrix (unnormalized)
%   xy - crosscorrelation between design matrix and dependent var (unnormalized)
%   wDims - [nt, nx] two-vector specifying # rows & # cols of w.
%   p - rank of bilinear filter
%   lambda - ridge parameter (optional)
%   opts - options struct (optional)
%
% Outputs:
%   wHat = estimate of full param vector
%   wt = column vectors
%   wx = row vectors

if (nargin >= 5) && ~isempty(lambda)  
    xx = xx + lambda*eye(size(xx)); % add ridge penalty to xx
end

% Set optimization options
if (nargin < 6) || isempty(opts)
    opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
    
    % OLD SYNTAX FOR OPTIONS:
    % opts = optimset('gradobj', 'on', 'Hessian', 'on','display','iter');
else
    opts = optimoptions(opts,'algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
    
    % OLD SYNTAX FOR OPTIONS:    
    %opts = optimset(opts, 'gradobj', 'on', 'Hessian', 'on');
end

% Set sizes and build sparse identity matrices
nt = wDims(1);
nx = wDims(2);
It = speye(nt);
Ix = speye(nx);

% Initialize estimate of w by linear regression and SVD
w0 = xx\xy;
[wt,s,wx] = svd(reshape(w0,nt,nx));
wt = wt(:,1:rnk)*sqrt(s(1:rnk,1:rnk));
wx = sqrt(s(1:rnk,1:rnk))*wx(:,1:rnk)';

prs0 = [wt(:); wx(:)];
floss = @(prs)(bilinRegressLoss(prs,nt,nx,rnk,xx,xy,Ix,It));
%HessCheck(floss,prs0+.1);

prs = fminunc(floss,prs0,opts);
wt = reshape(prs(1:nt*rnk),nt,rnk);
wx = reshape(prs((nt*rnk)+1:end),rnk,nx);
w_hat = wt*wx;


% ===========================================================
function [l,dl,H] = bilinRegressLoss(prs,nt,nx,rnk,C,b,Ix,It)
%
% Computes .5*w'*C*w - w'b and its derivatives, where w is parametrized as
% w = vec(wt*wx), giving a bilinear (low-rank) parametrization of the
% matrix wt*wx.


nnt = nt*rnk;
nnx = nx*rnk;
Wt = reshape(prs(1:nnt),nt,rnk);
Wx = reshape(prs(nnt+1:end),rnk,nx);

w = vec(Wt*Wx);
Mt = kron(Ix,Wt);
Mx = kron(Wx',It);

% Loss function
Cw = C*w;
l = .5*w'*Cw - w'*b;

% Gradient
dldt = Mx'*(Cw-b);
dldx = Mt'*(Cw-b);
dl = [dldt; dldx];

% Hessian
CMx = C*Mx;
Ha = Mx'*CMx;
Hb = Mt'*C*Mt;
Hba1 = Mt'*CMx;
Hba2 = (vecpermcols(kron(reshape(Cw-b,nt,nx)',eye(rnk)),nt,rnk));
Hba = Hba1+Hba2;
H = [Ha, Hba'; Hba, Hb];

