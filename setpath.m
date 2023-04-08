% setpath.m
%
% Script to set paths for multilinearRegression code

if ~exist('bilinearRegress_coordAscent_fast','file')  % check if bilinearRegress_coordAscent_fast is in path
    addpath fitting;
end

if ~exist('vecpermcols','file')  % check if function is in path
    addpath tools
end
