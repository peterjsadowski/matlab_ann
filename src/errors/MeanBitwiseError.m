function E = MeanBitwiseError(X, target)
% Compute mean bitwise zero-one error. 
% If values are real, we use X,target >= 0.5.
% INPUT: X = mxc matrix of binary or real values.
%        target = mxc matrix of binary or real values.
E = mean((target >= 0.5) ~= (X >= 0.5), 2);
end