function [E dE_dWW] = gradient_backgen(WW, arch, input, target)
% This function computes the error and gradient of a muliclass
% classification network with a softmax activation function in the last
% layer, and logistic units elsewhere.

% Compute activation of each layer.
W = vec2wcell(WW, arch);     % Each weight matrix is (nin+1) x nout.
nlayers = length(W); 
X    = cell(nlayers+1, 1);
X{1} = addbias(input);
for i = 1:nlayers-1
    % Logistic function, include bias.
    %X{i+1} = addbias(applylayer(W{i}, X{i}, 'logistic'));
    X{i+1} = addbias(1./(1+exp(-X{i} * W{i})));
end
% Softmax for output layer. Do not add bias.
X{nlayers+1} = exp(X{nlayers} * W{nlayers});
X{nlayers+1} = X{nlayers+1} ./ repmat(sum(X{nlayers+1},2), 1, size(X{nlayers+1},2));

% Cross entropy error for multiclass output. Different from crossent.
E_class = -sum(sum( target .* log(X{end}) )); % Might have to handle log(0).
dE_class = calcBackpropGradient(W, X, target); % cell array

% Remove bias terms.
for k = 1:nlayers
    X{k} = X{k}(:, 1:end-1);
end
% Compute backgen error and gradient.
[E_gen, dE_gen] = calcGenerativeError(W, X);
% Add bias terms.
for k = 1:nlayers
    dE_gen{k} = [dE_gen{k}; zeros(1, size(dE_gen{k}, 2))];
end

% Combine errors
lambda = 0.99;
E = lambda * E_class + (1-lambda) * E_gen;
for k = 1:nlayers
    dE{k} = lambda * dE_class{k} + (1-lambda) * dE_gen{k};
end

% E = E_class;
% dE = dE_class;
    
% Combine gradients into single vector, matching WW.
if size(dE,2)>1, dE = dE'; end; % Need a column cell. 
dE_dWW = cell2mat(cellfun(@(x) x(:), dE, 'UniformOutput',false)); % Column.
end

function X = addbias(X)
X = [X, ones(size(X,1),1)];
end

function dE = calcBackpropGradient(W, X, target)
% Compute backprop gradient from layer activations X and target.
% This is the same regardless of the activation in the last layer being
% crossent or softmax.
nlayers = length(W); 
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
    %
    dE{i} = X{i}' * dE_ds;
end
end

function [E, dE] = calcGenerativeError(W, X)
% Calculate generative crossentropy error and gradient.
nlayers = length(W);
E_k = zeros(nlayers,1);
Y = cell(nlayers, 1);
for k = 1:nlayers
    Y{k} = 1./(1+exp(-X{k+1} * W{k}(1:end-1,:)'));
    E_k(k) = sum(crossent(Y{k}, X{k}, false));
end
E = sum(E_k);
dE = calcGenerativeGradient(W, X, Y);
end

function dE = calcGenerativeGradient(W, X, Y)
% Calculate generative crossentropy gradient. 
nlayers = length(W);
m = size(X{1}, 1);
dE = cell(nlayers, 1);
for k = nlayers:-1:1
    
    % Common part
    commonsum = zeros(size(W{k}(1:end-1,:)));
    for i = 1:m
        temp1 = X{k}(i,:)' * (X{k+1}(i,:) .* (1-X{k+1}(i,:))); % rxs
        temp2 = ((Y{k}(i,:) - X{k}(i,:)) * W{k}(1:end-1,:)); % 1xs
        temp2 = repmat(temp2, size(Y{k},2), 1); % rxs
        commonsum = commonsum + (temp1 .* temp2);
    end
%     temp1 = X{k}' * (X{k+1} .* (1-X{k+1})); % rxs
%     temp2 = ((Y{k} - X{k}) * W{k}(1:end-1,:)); % mxs
%     temp2 = repmat(sum(temp2, 1), size(Y{k},2), 1); % rxs
        
    % Activation-specific sum
    if k == nlayers
        % Softmax activation in last layer, so different calculation.
        specificsum = zeros(size(W{k}(1:end-1,:)));
        for i = 1:m
            % (yr - xr) xs
            temp1 = (Y{k}(i,:) - X{k}(i,:))' * X{k+1}(i,:); % rxs
            % xr sum_i(wri xi)
            temp2b = X{k}(i,:)' .* (W{k}(1:end-1,:) * X{k+1}(i,:)');
            temp2b = repmat(temp2b, 1, size(W{k}, 2)); % rxs
            % xr xs wrs
            temp2c = X{k}(i,:)' * X{k+1}(i,:) .* W{k}(1:end-1,:); % rxs
            % Add contribution
            specificsum = specificsum + (temp1 .* (1 - temp2b + temp2c));
        end
    else
        % Crossent
        % dE/dwrs = sum_j (yj-xj) wjs (xs(1-xs)) xr + (yr-xr)xs
        specificsum = (Y{k} - X{k})' * X{k+1}; % rxs
    end
    dE{k} = commonsum + specificsum;
end
end