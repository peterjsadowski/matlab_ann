function [E dE] = gradient_regression(W, input, target)
% This function is computes the error and gradient of a network with
% sigmoidal transfer functions in all but the last layer, which is linear
% with sum of squares error.

% Compute activation of each layer.
% W = vec2wcell(WW, arch);     % Each weight matrix is (nin+1) x nout.
nlayers = length(W); 
X    = cell(nlayers+1, 1);
X{1} = addbias(input);
for i = 1:nlayers-1
    % Logistic function, include bias.
    %X{i+1} = addbias(applylayer(W{i}, X{i}, 'logistic'));
    X{i+1} = addbias(1./(1+exp(-X{i} * W{i})));
end
% Linear transfer function for output layer. Do not add bias.
X{nlayers+1} = X{nlayers} * W{nlayers}; % Linear.


% Cross entropy error for multiclass output. Different from crossent.
E = mean(sum( (target - X{end}).^2 , 2)); % Might have to handle log(0).

% Compute gradient.
% dE/dwji = (y_i - t_i)x_j for top layer weight.
dE_ds       = 2*(X{nlayers+1} - target);  % s (sum) is the input to the activation function.
dE{nlayers} = X{nlayers}' * dE_ds;
for i = nlayers-1:-1:1
    % dE/ds = dE/dx dx/ds         Where s is the input sum.
    % dE/dx = dE/dsprev dsprev/dx Calculated from layer above.
    % dx/ds = x(1-x)              Derivative of the logistic function.
    dE_ds = (dE_ds * W{i+1}') .* X{i+1} .* (1-X{i+1});
    % No input to bias unit.
    dE_ds = dE_ds(:,1:end-1);
    % Gradient
    dE{i} = X{i}' * dE_ds;
end


% % Combine gradients into single vector, matching WW.
% if size(dE,2)>1, dE = dE'; end; % Need a column cell. 
% dE_dWW = cell2mat(cellfun(@(x) x(:), dE, 'UniformOutput',false)); % Column.

% For reference.
% Ix3 = (Ix_class*w_class').*w3probs.*(1-w3probs);
% Ix3 = Ix3(:,1:end-1);
% dw3 =  w2probs'*Ix3;
% dE_dWW = [dw1(:)' dw2(:)' dw3(:)' dw_class(:)']'; 
end

function X = addbias(X)
X = [X, ones(size(X,1),1)];
end
