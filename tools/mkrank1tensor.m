function M = mkrank1tensor(x1,x2,x3)
% M = mkrank1tensor(x1,x2,x3)
%
% Constructs a 3rd order rank-1 tensor out of the column vectors x1, x2, x3

M = pagemtimes(x1*x2',permute(x3,[2 3 1]));