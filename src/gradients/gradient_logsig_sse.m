function [E dE] = gradient_logsig_sse(W, input, target)
% This function is computes the error and gradient of a network with 
% logistic transfer functions and sum of squares error.
% This is used for learning deep target layers, in which we have logistic
% transfer but we want a simple error function.

% Compute activation of each layer.
nlayers = length(W); 
X    = cell(nlayers+1, 1);
X{1} = addbias(input);
for i = 1:nlayers-1
    X{i+1} = addbias(1./(1+exp(-X{i} * W{i})));
end
X{nlayers+1} = 1./(1+exp(-X{nlayers} * W{nlayers})); % Output, no bias.

% Sum of squares error function.
E = mean(sum( (X{end} - target).^2 , 2));

% Compute gradient.
% dE/dwji = 2*(y_i - t_i) * (y_i (1-y_i)) * x_j for top layer weight.
dE_ds = 2*(X{nlayers+1} - target) .* X{nlayers+1} .* (1 - X{nlayers+1});
% dE_ds       = X{nlayers+1} - target;  % s (sum) is the input to the activation function.
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

function X = addbias(X)
X = [X, ones(size(X,1),1)];
end
