function [E dE] = gradient_threshold(W, input, target, arrayflag)
% This function is computes the error and gradient of a single layer 
% perceptron with threshold gate transfer function -> {0,1} and Hamming error.

if iscell(W)
    nlayers = length(W); assert(nlayers == 1);
    W = W{1};
end
m = size(input, 1);

% For 0,1 setup, dE/dwkj ~ (bk - data)*bj in perceptron learning algorithm.
X1 = [input, ones(m,1)];
X2 = (X1 * W) > 0; % mxn.
dE_dx2  = X2 - target;         % mxn2
dE{1}   = X1' * dE_dx2;            % n1xn2

if nargin > 3 && arrayflag
    % Return mean error for each output (column).
    E = biterr(target, X2, 'column-wise') / size(target,1);
else
    % Mean over all samples (rows), outputs (columns).
    E = mean(biterr(target, X2, 'row-wise'));
end
    
end
