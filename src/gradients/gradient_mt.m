function [E, dE] = gradient_mt(W, data, alpha)
% This function computes the error E and gradient dE of a specific architecture:
% -The lower layers are all feedforward with logsig transfer function.
% -The last layer is a multitask architecture with 2 tasks:
%   1) Classification output with logsig/SoftmaxTransfer and 
%      CrossEntropyError/MulticlassCrossEntropyError
%   2) Autoencoder output nodes with logsig transfer and CrossEntropyError.
%  The error is a weighted sum (1-alpha)*Error1 + alpha*Error2.
%  The gradient is thus a linear combination of grad1, grad2; we compute
%  the combination rather than the individual gradients for speed.
%  
% INPUT: 
%   W = Cell array of weight matrices. (Single arg used by minimize.m)
%       W(1:end-2) = Weights for lower layers.
%       W(end-1)   = Weights for task1 classify output.
%       W(end)     = Weigths for task2 autoencode output.
%  data = struct with following fields:
%         input   = m x nfeat data matrix
%         target  = m x ntarget data matrix. Target for W_classify.
%         <target2> = m x ntarget2 data matrix. Target for W_autoencode.(usually = input)
%         <unlabeled> = mxnfeat data matrix. Unlabeled data for task2.
%  alpha = Parameter in [0,1]. Specifies weight for task 2. 

% Unpack input.
W_classify   = W{end-1};
W_autoencode = W{end};
W            = W(1:end-2);
nlayers      = length(W) + 1;

if ~isfield(data, 'target2')
    % Default: autoencode target (target2) is the input. 
    data.target2 = data.input;
end
if isfield(data, 'unlabeled') && (alpha>0) && ~isfield(data, 'targetunlabeled')
    % Default: target for unlabeled data is unlabeled.
    data.targetunlabeled = data.unlabeled;
end

% Forward pass: Compute activation of each layer.
[X, X_classify] = calcClassActivations(W, W_classify, data.input);

% Compute error E. This is the objective function -- it should decrease.
[E_classify] = calcClassError(X_classify, data.target);

% Compute gradient for top layer weights.
% dE/dwji = (y_i - t_i)x_j for top layer weight. 
% This holds for both crossent and softmax crossent.
dE_ds_classify  = (X_classify - data.target) / size(data.target,1);  % s (sum) is the input to the activation function.
dE_classify_out = X{nlayers}' * dE_ds_classify;
dE_classify     = calcGradBackpropLogsig(vertcat(W,W_classify), X, dE_ds_classify);

% % Sanity check
% [EE dEE] = gradient_softmaxcrossent(vertcat(W, W_classify), data.input, data.target);
% keyboard
% assert(sum(sum(abs(dEE{1} - dE_classify{1}*size(data.target,1)))) < 1e-6);
% assert(sum(sum(abs(dEE{end} - dE_classify_out*size(data.target,1)))) < 1e-6);

if alpha == 0
    % We are done.
    E = E_classify;
    dE = cell(nlayers+1, 1);
    dE(1:end-2) = dE_classify;
    dE{end-1} = dE_classify_out;
    dE{end}   = zeros(size(W_autoencode));
    return
end

% Logsig for output 2. Do not add bias.
X_autoencode = 1./(1+exp(-X{end} * W_autoencode));
% Autoencode error: MEAN over all examples .
%                   MEAN over all outputs because this makes weighing
%                   against the classification task easier. 
% Note: CrossEntropyError sums over outputs, so need to divide by #out.
E_autoencode = mean(CrossEntropyError(X_autoencode, data.target2)) / size(data.target2,2);
% dE/dW_autoencode needs some additional factors.
dE_ds_autoencode  = (X_autoencode - data.target2) / numel(data.target2);  % s (sum) is the input to the activation function.
dE_autoencode_out = X{nlayers}' * dE_ds_autoencode;
dE_autoencode     = calcGradBackpropLogsig(vertcat(W, W_autoencode), X, dE_ds_autoencode);

% Backprop together.
% dE_ds_classify  = (X_classify - data.target) * (1-alpha);  % s (sum) is the input to the activation function.
% dE_classify_out = X{nlayers}' * dE_ds_classify;
% dE_ds_autoencode  = (X_autoencode - data.target2) / size(data.target2,2) * alpha;  % s (sum) is the input to the activation function.
% dE_autoencode_out = X{nlayers}' * dE_ds_autoencode;
% dE_ds = [dE_ds_classify, dE_ds_autoencode];
% assert(length(W) == nlayers-1)
% % Standard backprop on logsig transfers.
% dE = cell(nlayers+1, 1);
% dE(1:end-2) = calcGradBackpropLogsig(vertcat(W,[W_classify, W_autoencode]), X, dE_ds);
% dE{end-1} = dE_classify;
% dE{end} = dE_autoencode;

% Optional: unsupervised data
if isfield(data, 'unlabeled') && (alpha > 0)
    assert((numel(data.input) == numel(data.target2)) == (numel(data.unlabeled) == numel(data.targetunlabeled)))
    
    [E_autoencode2, dE_autoencode2, dE_autoencode2_out] = calcUnlabeledGrad(W, W_autoencode, data.unlabeled, data.targetunlabeled);
    
    % Beta is the weight assigned to the unlabeled data. (Depends on #samples.)
    beta = size(data.unlabeled,1) / (size(data.unlabeled,1) + size(data.input,1));
    % Autoencoder gradient is a weighted sum.
    E_autoencode = (1-beta) * E_autoencode + beta * E_autoencode2;
    for i = 1:nlayers-1
        dE_autoencode{i} = (1-beta) * dE_autoencode{i} + beta * dE_autoencode2{i};
    end
    dE_autoencode_out = (1-beta) * dE_autoencode_out + beta * dE_autoencode2_out;
end

% Total error is a weighted sum. 
E = (1-alpha) * E_classify + alpha * E_autoencode;

% Total gradient is a weighted sum.
dE = cell(nlayers+1, 1);
for i = 1:nlayers-1
    dE{i} = (1-alpha) * dE_classify{i} + alpha * dE_autoencode{i};
end
dE{end-1} = (1-alpha) * dE_classify_out;
dE{end}   = alpha * dE_autoencode_out;

end

function [E, dE, dE_out] = calcUnlabeledGrad(W, W_autoencode, input, target)
    % Calculate unsupervised gradient on unlabeled data. 
    % Update dE by adding this gradient, scaled appropriately (by alpha and noutputs).
    % Note: In our error and gradient, we MEAN over examples, and MEAN over
    %       output units. Thus lots of unlabeled data will result in more
    %       emphasis on the autoencoder task. 
    % Use gradient_crossent function, but need to scale by #outputs.
    [E, dE] = gradient_crossent(vertcat(W, W_autoencode), input, target);
    num = numel(target);    
    E = E / num;
    for i = 1:length(dE)
        dE{i} =  dE{i} / num;
    end
    
    dE_out = dE{end};
    dE = dE(1:end-1);
end

function dE = calcGradBackpropLogsig(W, X, dE_ds)
% Calculate the gradient of each layer in a mlp of logsig transfer
% functions via backpropagation.
% INPUT: 
%   W = Cell array of weights of length nlayers
%   X = Cell array of activations of length nlayers (but includes input)
%   dE_ds = Derivative of the objective w.r.t. the output sum s, which is
%           the input to the output neurons before the transfer function is
%           applied.
nlayers = length(W);
dE = cell(nlayers-1,1);
for i = nlayers-1:-1:1
    % dE/ds = dE/dx dx/ds         Where s is the input sum.
    % dE/dx = dE/dsprev dsprev/dx Calculated from layer above.
    % dx/ds = x(1-x)              Derivative of the logistic function.
    dE_ds = (dE_ds * W{i+1}') .* X{i+1} .* (1-X{i+1});
    % No input to bias unit.
    dE_ds = dE_ds(:,1:end-1); 
    dE{i} = X{i}' * dE_ds;
end
end

function [E_classify, E_autoencode] = calcError(X_classify, target1, X_autoencode, target2)
% E            = (1-alpha)*E_classify + alpha*E_autoencode
% Classification error: MEAN over all examples.
%                       MEAN over all outputs because softmax output should
%                       be summed, not averaged, when comparing to other
%                       cross-entropy errors.
if size(target1,2) == 1
    E_classify = mean(CrossEntropyError(X_classify, target1));
else
    E_classify = mean(MulticlassCrossEntropyError(X_classify, target1));
end

% Autoencode error: MEAN over all examples .
%                   MEAN over all outputs because this makes weighing
%                   against the classification task easier. 
% Note: CrossEntropyError sums over outputs, so need to divide by #out.
E_autoencode = mean(CrossEntropyError(X_autoencode, target2)) / size(target2,2);
end

function [E_classify, E_autoencode] = calcClassError(X_classify, target1)
% E            = (1-alpha)*E_classify + alpha*E_autoencode
% Classification error: MEAN over all examples.
%                       MEAN over all outputs because softmax output should
%                       be summed, not averaged, when comparing to other
%                       cross-entropy errors.
if size(target1,2) == 1
    E_classify = mean(CrossEntropyError(X_classify, target1));
else
    E_classify = mean(MulticlassCrossEntropyError(X_classify, target1));
end
end

function [X, X_classify, X_autoencode] = calcAllActivations(W, W_classify, W_autoencode, input)
% Compute activation of each layer and 2 separate output layers. Used in grad calc.
X = calcActivitions(W, input);
% Softmax for output 1. Do not add bias.
if size(W_classify,2) == 1
    % Twoclass problem.
    X_classify = 1./(1+exp(-X{end} * W_classify));
else
    % Multiclass problem.
    X_classify = SoftmaxTransfer(X{end} * W_classify);
end
% Logsig for output 2. Do not add bias.
X_autoencode = 1./(1+exp(-X{end} * W_autoencode));
end

function [X, X_classify] = calcClassActivations(W, W_classify, input)
% Compute activation of each layer and 2 separate output layers. Used in grad calc.
X = calcActivitions(W, input);
% Softmax for output 1. Do not add bias.
if size(W_classify,2) == 1
    % Twoclass problem.
    X_classify = 1./(1+exp(-X{end} * W_classify));
else
    % Multiclass problem.
    X_classify = SoftmaxTransfer(X{end} * W_classify);
end
end

function X = calcActivitions(W, input)
% Compute activation of each lower layer. Used in grad calc.
nlayers = length(W) + 1;
X    = cell(nlayers, 1); % Input layer, then hidden.
X{1} = addbias(input);
for i = 1:nlayers-1
    X{i+1} = addbias(1./(1+exp(-X{i} * W{i}))); % logsig (explicit for speed)
end
end
