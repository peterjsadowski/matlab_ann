classdef perceptron
    properties
        nu = 1;          % Learning rate. Used for netdt.
        mu = 0;          % Weight decay, L2 penalty. Used for netdt. 
        momentum = 0;    % Momentum term. 
        
        TransferFcn     % String specifying name of transfer function e.g. logsig
        initscale = 1   % Initial scale of weights. (Var of sum)
        
        % For training
        ErrorFcn = []

    end
    properties (Dependent = true)
        W               % Weight matrix
        NumInputs       % Number of inputs, not including bias.
        NumOutputs
        GradFcn   
        
        fullyconnected  % Is the layer fully connected.
        
    end
    properties (Access = 'private')
        weights = [];      % Accessed only through dependent property W.
        connectivity = []; % Connectivity between layers. If empty, assume full.
        dW = [];           % Remember last weight update for momentum.
        
    end   
        
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        % Constructor
        function self = perceptron(NumInputs, NumOutputs, TransferFcn, ErrorFcn)
            self.TransferFcn = TransferFcn;
            if nargin>3, self.ErrorFcn = ErrorFcn; end;
            self.W = zeros(NumInputs+1, NumOutputs); % Specifies arch.
            self = initialize(self);
        end
        % Modifications
        function self = initialize(self, connectivity)
            % Initialize weights so that sum has var initscale.
            % Initialize connectivity matrix if specified.
            if nargin == 1
                % Full connectivity.
                self.connectivity = [];
                self.W = [(2*rand(self.NumInputs, self.NumOutputs) - 1); zeros(1, self.NumOutputs)];
                self.W = self.initscale * sqrt(1/(self.NumInputs+1)) * self.W;
            else
                % Use specified connectivity, or create new connectivity matrix.
                if length(connectivity) == 1
                    self.connectivity = createConnectivity(self.NumInputs, self.NumOutputs, connectivity);
                else
                    % Connectivity is a matrix.
                    self.connectivity = connectivity;
                end
                % Create random weight matrix.
                nin = self.NumInputs;
                nout = self.NumOutputs;
                self.W = [(2*rand(nin, nout) - 1); zeros(1, nout)];
                % Rescale weights.
                scale = self.initscale * sqrt(1 ./ sum(self.connectivity));
                scale = repmat(scale, self.NumInputs+1, 1);
                self.W = self.W .* scale;
            end
            % Initialize dW
            self.dW = zeros(size(self.W));
        end
        function self = initializeBias(self, bias)
            % Set bias of all nodes. By default this is 0.
            self.W(end, :) = bias; 
        end
        function self = pretrain_rbm(self, data, nepoch)
            % Use contrastive divergence to train weights on this input.
            r = rbm(self.NumInputs, self.NumOutputs);
            r = r.train(data, nepoch);
            self.W = r.W;
        end
        % Properties
        function NumInputs = get.NumInputs(self)
            NumInputs = size(self.W, 1) - 1;
        end
        function NumOutputs = get.NumOutputs(self)
            NumOutputs = size(self.W, 2);
        end
        function GradFcn = get.GradFcn(self)
            % Determines appropriate gradient function from TransferFcn and ErrorFcn.
            if strcmpi(self.TransferFcn, 'logsig') && strcmpi(self.ErrorFcn, 'CrossEntropyError')
                GradFcn = 'gradient_crossent';
            elseif strcmpi(self.TransferFcn, 'SoftmaxTransfer') && strcmpi(self.ErrorFcn, 'MulticlassCrossEntropyError')
                % SoftmaxTransfer transfer function, for multi class classification.
                GradFcn = 'gradient_softmaxcrossent';
            elseif strcmpi(self.TransferFcn, 'linear') && strcmpi(self.ErrorFcn, 'SumSquaredError')
                % Linear transfer function and SSE.
                GradFcn = 'gradient_regression';
            elseif strcmpi(self.TransferFcn, 'threshold') && strcmpi(self.ErrorFcn, 'Hamming')
                % Perceptron learning approx gradient for threshold transfer function and mean Hamming error.
                GradFcn = 'gradient_threshold';
            elseif strcmpi(self.TransferFcn, 'logsig') && strcmpi(self.ErrorFcn, 'SumSquaredError')
                GradFcn = 'gradient_logsig_sse';
            elseif strcmpi(self.TransferFcn, 'logsig') && strcmpi(self.ErrorFcn, 'L1Error')
                GradFcn = 'gradient_logsig_l1';
            else
                %GradFcn = [];
                error('Unknown transfer-error pair.');
            end
        end
        function fullyconnected = get.fullyconnected(self)
            % Returns true if all outputs connected to all inputs.
            fullyconnected = isempty(self.connectivity) || all(all(self.connectivity));
        end   
        function W = get.W(self)
            W = self.weights;
        end
        function self = set.W(self, W)
            % Sets W, while enforcing limited connectivity.
            if isempty(self.connectivity)
                % Fully connected.
                self.weights = W;
            else
                self.weights = W .* self.connectivity;
            end
        end 
        % Calculations
        function output = apply(self, input)
            assert(~strcmp(self.TransferFcn, 'softmax'), 'Use SoftmaxTransfer instead.');
            output = feval(self.TransferFcn, [input, ones(size(input, 1), 1)] * self.W);
        end
        function output = apply_sigmoid_stochastic(self, input)
           % Apply weights, sigmoidal transfer, and random sample to get 
           % stochastic output.
           output = feval('logsig', [input, ones(size(input, 1), 1)] * self.W);
           output = rand(size(output)) < output;
        end
        function input = apply_sigmoid_stochastic_reverse(self, output)
           % Apply weights, sigmoidal transfer, and random sample to get 
           % stochastic sample from visible layer conditioned on hidden.
           % Ignores bias term.
           input = feval('logsig', output * self.W(1:end-1,:)');
           input = rand(size(input)) < input;
        end
        function error = test(self, data)
            % Compute error given input and target.
            error = feval(self.ErrorFcn, apply(self, data.input), data.target);
        end
        function nu = calcAdjustednu(self, nu)
            % Scale the learning rate by the number of inputs to avg output
            if self.fullyconnected
                nu = nu / (self.NumInputs + 1); % Include bias term.
            else
                nu = nu / mean(sum(self.connectivity));
            end
        end
        % Training functions
        function self = train_gd(self, data, niter)
            % Perform simple weight updates to this single layer perceptron.
            for i = 1:niter
                % Passes W in form of a cell array.
                [~, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target);
                dE_dW = dE_dW{1}; % Single layer.
                % Scale so that dE is average per sample.
                dE_dW = dE_dW / size(data.input, 1);
                
                % Momentum
                %self.dW_prev = self.dW;
                self.dW = - (self.nu * dE_dW);
                %self.dW = - (self.nu * dE_dW) + (self.momentum * self.dW);
                %assert(self.fullyconnected)

                % Shrink dW as niter.
                %self.dW = self.dW * ((niter + 1 - i) / niter);
                
                % Update.
                self.W = self.W + self.dW;   % Not ready - (self.mu * self.W);
                
                if all(self.dW == 0)
                    break % Done training.
                end
            end
        end
        function self = train_gd_min(self, data, maxstep)
            % Train using conjugate gradient search.
            if strcmp(self.GradFcn, 'gradient_threshold')
                % Use gradient_crossent because gradient_theshold will take 0 step.
                gradfcn = 'gradient_crossent';
            else
                gradfcn = self.GradFcn;
            end
            [W, fX, i] = minimize({self.W}, gradfcn, maxstep, true, data.input, data.target);
            self.W = W{1};
        end
        function self = train_gd_adapt(self, data, maxiter)
            % Use adaptive learning rate, maxiter. 
            nu_adapt = self.nu; % Adaptive learning rate initialized.
            E_prev = inf;
            for i = 1:maxiter
                % Compute error and gradient.
                [E, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target);% Passes W in form of a cell array.
                dE_dW = dE_dW{1}; % Single layer.
                assert(~any(isnan(dE_dW(:))));
                assert(~isnan(E));
                
                % Update adaptive learning rate.
                if E == 0
                    return % Done training.
                elseif E > E_prev
                    nu_adapt = max(nu_adapt * 0.7, 1e-6);
                else
                    nu_adapt = min(nu_adapt * 1.2, self.nu);
                end
                E_prev = E;
                
                % Scale so that dE is average per sample. 
                dE_dW = dE_dW / size(data.input, 1);
                % Scaled to account for number of inputs. dE_dW = dE_dW / size(data.input, 2);
                %dE_dW = dE_dW / numel(data.input); % This could be 1e-6
                
                % Update.
                self.dW = - (nu_adapt * dE_dW);
                self.W = self.W + self.dW; 
                % Weight decay
                %self.W = self.W - (sign(self.W) .* min(abs(self.W), self.mu));

                assert(~any(isnan(self.dW(:))))
                assert(~any(isnan(self.W(:))))
            end
        end
        function self = train_gd_ipocket(self, data, maxiter)
            % Use constant learning rate, maxiter. 
            % This method assumes independent outputs.
            if strcmp(self.GradFcn, 'gradient_softmaxcrossent')
                %self = train_gd_min(self, data, 100);
                self = train_gd(self, data, maxiter);
                return
            end
            assert(~strcmp(self.GradFcn, 'gradient_softmaxcrossent'), 'Softmax transfer cannot be used with indep-pocket method.');
            assert(strcmp(self.GradFcn, 'gradient_threshold') ...
                   || strcmp(self.GradFcn, 'gradient_crossent'), ...
                   'Need to implement arrayflag option in other gradfcns');
                
            % Keep track of minimum Error, Weights for each output.
            E_min  = inf * ones(1, size(self.W, 2));
            W_min  = self.W;
            E_min_hist = zeros(1, maxiter);
            nwindow = 50;
            for i = 1:maxiter+1 % Because we perform a loop with initial w.
                % Compute error and gradient, with arrayflag to get columns
                [E, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target, true);% Passes W in form of a cell array.
                dE_dW = dE_dW{1}; % Single layer.
                assert(~any(isnan(dE_dW(:))));
                assert(~any(isnan(E)));
                
                % Keep best weights in pocket.
                W_min(:,E<=E_min) = self.W(:, E<=E_min);
                E_min(E<=E_min) = E(E<=E_min);
                E_min_hist(i) = sum(E_min); % History of E_min
                
                % Update adaptive learning rate.
                if sum(E_min_hist(i)) == 0
                    self.W = W_min;
                    return % Done training.
                elseif (i>nwindow) && (E_min_hist(i-nwindow) == E_min_hist(i))
                    % Slow progress. End early.
                    warning('Ending training early due to slow progress')
                    self.W = W_min;
                    return 
                end
                    
                % Scale so that dE is average per sample. 
                dE_dW = dE_dW / size(data.input, 1);
                % Scaled to account for number of inputs. dE_dW = dE_dW / size(data.input, 2);
                %dE_dW = dE_dW / numel(data.input); % This could be 1e-6
                
                % Update.
                self.dW = - (self.nu * dE_dW);
                self.W = self.W + self.dW; 
                % Weight decay
                %self.W = self.W - (sign(self.W) .* min(abs(self.W), self.mu));

                assert(~any(isnan(self.dW(:))))
                assert(~any(isnan(self.W(:))))
            end
            self.W = W_min;
            %plot(E_hist); hold on; plot(E_min_hist, 'r'); hold off; pause;
        end
        function self = train_gd_adapt_pocket(self, data, maxiter)
            % Use adaptive learning rate, maxiter. 
            nu_adapt = self.nu; % Adaptive learning rate initialized.
            E_prev = inf;
            E_min  = inf;
            W_min  = self.W;
            for i = 1:maxiter
                % Compute error and gradient.
                [E, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target);% Passes W in form of a cell array.
                dE_dW = dE_dW{1}; % Single layer.
                assert(~any(isnan(dE_dW(:))));
                assert(~isnan(E));
                
                % Keep best weights in pocket.
                if E < E_min
                    W_min = self.W;
                    E_min = E;
                end
                
                % Update adaptive learning rate.
                if E == 0
                    return % Done training.
                elseif E > E_prev
                    nu_adapt = max(nu_adapt * 0.7, 1e-6);
                else
                    nu_adapt = min(nu_adapt * 1.2, self.nu);
                end
                E_prev = E;
                
                % Scale so that dE is average per sample. 
                dE_dW = dE_dW / size(data.input, 1);
                % Scaled to account for number of inputs. dE_dW = dE_dW / size(data.input, 2);
                %dE_dW = dE_dW / numel(data.input); % This could be 1e-6
                
                % Update.
                self.dW = - (nu_adapt * dE_dW);
                self.W = self.W + self.dW; 
                % Weight decay
                %self.W = self.W - (sign(self.W) .* min(abs(self.W), self.mu));

                assert(~any(isnan(self.dW(:))))
                assert(~any(isnan(self.W(:))))
            end
            self.W = W_min;
        end
        function self = train_gd_adapt_cautious(self, data, maxiter)
            % Use adaptive learning rate, maxiter. 
            nu_adapt = self.nu; % Adaptive learning rate initialized.
            E_prev = inf;
            for i = 1:maxiter
                % Compute error and gradient.
                [E, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target);% Passes W in form of a cell array.
                dE_dW = dE_dW{1}; % Single layer.
                assert(~any(isnan(dE_dW(:))));
                assert(~isnan(E));
                
                % Ensure progress, update adaptive learning rate.
                if E == 0
                    % Done training.
                    return 
                elseif E > E_prev
                    % Forget previous step. Try along same gradient with shorter step.
                    self.W = self.W - self.dW * 0.3; % Retract a portion of the step that led us astray.
                    self.dW = self.dW * 0.7; % This is the dW we meant to make.
                    nu_adapt = min(nu_adapt * 0.7, 1e-6); % Corresponding nu.
                    continue
                else
                    nu_adapt = max(nu_adapt * 1.2, self.nu);
                end
                E_prev = E;
                
                % Scale so that dE is average per sample. 
                dE_dW = dE_dW / size(data.input, 1);
                % Scaled to account for number of inputs. dE_dW = dE_dW / size(data.input, 2);
                %dE_dW = dE_dW / numel(data.input); % This could be 1e-6
                
                % Update.
                self.dW = - (nu_adapt * dE_dW);
                self.W = self.W + self.dW; 
                % Weight decay
                %self.W = self.W - (sign(self.W) .* min(abs(self.W), self.mu));

                assert(~any(isnan(self.dW(:))))
                assert(~any(isnan(self.W(:))))
            end
            %warning('Didnt finish training layer')
            %keyboard
        end
        function self = train_gd_dropout(self, data, niter, notdropidx)
            % Weight updates non-dropped perceptron outputs.
            % Used in netdt_dropout.
            for i = 1:niter
                % Passes W in form of a cell array.
                [~, dE_dW] = feval(self.GradFcn, ...
                    {self.W(:, notdropidx)}, ...
                    data.input, data.target(:, notdropidx));
                dE_dW = dE_dW{1}; % Single layer.
                
                % Scale so that dE is average per sample.
                dE_dW = dE_dW / size(data.input, 1);
                
                % No gradient for bias term either.
                dE_dW(end, :) = 0;
                
                % Update W
                assert(self.momentum == 0, 'Momentum not implemented.');
                self.W(:, notdropidx) = self.W(:, notdropidx) - (self.nu * dE_dW);
                %self.dW = - (self.nu * dE_dW);
                %self.W = self.W + self.dW;   % Not ready - (self.mu * self.W);
                
                if all(self.dW == 0)
                    break % Done training.
                end
            end
        end
        function self = train_gd_nobias(self, data, niter)
            % Perform simple weight updates to this single layer perceptron.
            for i = 1:niter
                % Passes W in form of a cell array.
                [~, dE_dW] = feval(self.GradFcn, {self.W}, data.input, data.target);
                dE_dW = dE_dW{1}; % Single layer.
                % Scale so that dE is average per sample.
                dE_dW = dE_dW / size(data.input, 1);
                % No bias gradient.
                dE_dW(end,:) = 0;
                % Momentum
                %self.dW_prev = self.dW;
                self.dW = - (self.nu * dE_dW) - (self.mu * self.W) + (self.momentum * self.dW);
                %assert(self.fullyconnected)
                % Update.
                self.W = self.W + self.dW;   % Not ready - (self.mu * self.W);
                
            end
        end
        function self = train_regression(self, data, niter)
            % Use linear regression to get weights.
            [m, nout] = size(data.target);
            for i = 1:nout
                self.W(:,i) = regress(data.target(:,i), [data.input, ones(m,1)]);
            end
        end
        function self = train_gd_crossent(self, data, niter)
            % Update weights assuming logsig transfer and crossentropy error.
            for i = 1:niter
                % Passes W in form of a cell array.
                [~, dE_dW] = feval('gradient_crossent', {self.W}, data.input, data.target);
                dE_dW = dE_dW{1}; % Single layer.
                % Scale so that dE is average per sample.
                dE_dW = dE_dW / size(data.input, 1);
                self.dW = - (self.nu * dE_dW);
                % Update.
                self.W = self.W + self.dW;   % Not ready - (self.mu * self.W);
                
                if all(self.dW == 0)
                    break % Done training.
                end
            end
        end
           
    end
  
    % ==================================================
    % PRIVATE METHODS
    % ==================================================
    methods (Access='private')
        
    end
end

% ==================================================
% OTHER METHODS
% ==================================================
function connectivity = createConnectivity(nin, nout, ncon)
% Create nin x nout matrix, where each output is connected to
% ncon inputs, plus the bias term.
if (ncon > (nin/nout))
    % Handle this better.
    warning('Connections will not be evenly distributed.');
    fprintf('Warning: Inputs: %d, Outputs: %d, Connectivity: %d\n', nin, nout, ncon);
end
connectivity = zeros(nin+1, nout);
for j = 1:nout
    idx1 = floor((nin / nout)*(j-1)) + 1;
    idx = idx1 : min(idx1 + ncon - 1, nin);
    connectivity(idx, j) = 1; % Ncon terms.
    connectivity(end, j) = 1; % Bias term.
end
end