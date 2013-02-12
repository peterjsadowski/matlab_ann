function [E dE] = gradient_crossent(W, input, target, arrayflag)
% This function is computes the error and gradient of a network with a 
% logistic activation function and summed cross entropy error.
% INPUT: 
%   W = cell array of length nlayers. Each layer is ninput+1 x nout matrix.
%   input = m x nfeat data matrix
%   target =  m x ntarget data matrix

% Compute activation of each layer.
nlayers = length(W); 
X       = cell(nlayers+1, 1);
X{1}    = addbias(input);
for i = 1:nlayers-1
    X{i+1} = addbias(1./(1+exp(-X{i} * W{i})));
end
X{nlayers+1} = 1./(1+exp(-X{nlayers} * W{nlayers})); % Output, no bias.

% Cross entropy error for independent outputs output.
%E = -sum(sum( target .* log(X{end}) + (1-target) .* log(1-X{end}) ));
if nargin > 3 && arrayflag
    % Return sum error over each output (column).
    E = -sum(target.*log(X{end}) + (1-target).*log(1-X{end}), 1);
    if any(isnan(E))
        % Machine precision hack.
        epsilon = 1e-45; % To avoid log(1e-46)=-inf
        E = -sum(target.*log(max(X{end}, epsilon)) + (1-target).*log(max(1-X{end}, epsilon)), 1);
    end
else
    % Return sum error over each sample (row) and output (column).
    E = sum(CrossEntropyError(X{end}, target)); % Handles log(0).
end

% Compute gradient.
% dE/dwji = (y_i - t_i)x_j for top layer weight.
dE_ds       = X{nlayers+1} - target;  % s (sum) is the input to the activation function.
dE{nlayers} = X{nlayers}' * dE_ds;
for i = nlayers-1:-1:1
    % dE/ds = dE/dx dx/ds         Where s is the input sum.
    % dE/dx = dE/dsprev dsprev/dx Calculated from layer above.
    % dx/ds = x(1-x)              Derivative of the logistic function.
    dE_ds = (dE_ds * W{i+1}') .* X{i+1} .* (1-X{i+1});
    % No input to bias unit.
    dE_ds = dE_ds(:,1:end-1); 
    dE{i} = X{i}' * dE_ds;
end

% % Combine gradients into single vector, matching WW.
% if size(dE,2)>1, dE = dE'; end; % Need a column cell. 
% dE_dWW = cell2mat(cellfun(@(x) x(:), dE, 'UniformOutput',false)); % Column.
end
