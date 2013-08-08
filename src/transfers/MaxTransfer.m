function Y = MaxTransfer(X)
% This max function gives a single 1 output (the max sum), and the rest 0.
% i.e. winner takes all. 
% Y = max(X')'; % MATLAB's softmax sometimes returns NaNs

% X is matrix of ninputs, noutputs.
[~,idx] = max(X, [], 2);

linidx = sub2ind(size(X), (1:size(X,1))', idx); 

Y = zeros(size(X));
Y(linidx) = 1;

assert(all(sum(Y,2) == 1))

if any(isnan(Y(:)))
    keyboard
end

end