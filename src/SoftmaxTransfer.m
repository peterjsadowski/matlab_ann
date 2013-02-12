function Y = SoftmaxTransfer(X)
% This softmax function overwrites the MATLAB default, which normalizes by
% column.
% Y = softmax(X')'; % MATLAB's softmax sometimes returns NaNs

epsilon = 1e-32;
X = exp(min(X, 20)) + epsilon;
Y = X ./ repmat(sum(X, 2), 1, size(X, 2));

if any(isnan(Y(:)))
    keyboard
end

end