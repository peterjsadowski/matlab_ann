function error = Hamming(X, target)
% Hamming distance.
error = biterr(X, target, 'row-wise');
end