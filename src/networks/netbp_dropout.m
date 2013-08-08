classdef netbp_dropout < netbp
% Net trained with backpropagation and dropout.
    properties
        nu = 0.1; % Learning rate.
        mu = 0.0; % Momentum
        dW = {};  % Last parameter step. (For momentum).
    end
    properties (Dependent = true)
        l2penalty        % L2 penalty on weights, weight decay
        dropprob  % Probability of dropping node in each layer.
        % Analysis
    end
    properties (Access = 'private')
        l2penalty_vec      % l2penalty
        dropprob_vec
    end
    methods
        function self = netbp_dropout(varargin)
            self = self@netbp(varargin{:});
            self.dropprob = 0;
        end
        function self = train(self, data)
            % Train network.
            if ischar(data), self.dataset=data; data = loaddata(data); end;
            batchdata = makeBatchData(data, self.batchsize);
            nbatch = length(batchdata);
            % Record initial error.
            if isempty(self.ErrorTrain)
                self = recordErrors(self, data);
                printstatus(self);
            end
            % Initialize momentum.
            for k = 1:self.nlayers
                self.dW{k} = zeros(size(self.W{k}));
            end
            % Begin learning
            for i = 1:self.nepoch
                for j = 1:nbatch
                    % Compute gradient.
                    [E dE_dW] = gradient_dropout(self, batchdata{j});
                    % Update weights.
                    for k = 1:self.nlayers
                        % Scale so that dE is average per sample.
                        % dW = mu*dW + (1-mu) * nu * (-grad)
                        self.dW{k} = (self.dW{k} * self.mu) ...
                                     - (dE_dW{k} * (self.nu * (1 - self.mu) / size(data.input, 1))); 
                        self.W{k} = self.W{k}  + self.dW{k};
                    end
                end % batch
                % Record errors
                self = recordErrors(self, data);
                printstatus(self)
            end % epoch
        end
        % Properties
        function self = set.dropprob(self,  dropprobs)
            % Set dropprob of units in each layer.
            if length(dropprobs) == 1
                self.dropprob_vec = dropprobs * ones(1, self.nlayers);
            else
                assert(length(dropprobs) == self.nlayers);
                self.dropprob_vec = dropprobs;
            end
        end
        function dropprob = get.dropprob(self)
            dropprob = self.dropprob_vec;
        end
        % Computations
        function dropped = sampledrop(self)
            % Randomly assign nodes to drop.
            dropped = cell(1, self.nlayers);
            for i = 1:self.nlayers
                % Bias can be dropped too.
                dropped{i} = rand(1, self.layers{i}.NumInputs + 1) < self.dropprob(i);
            end
        end
        function W_dropped = dropoutweights(self, dropped)
           % Return Weights that are equivalent to dropping nodes.
           W_dropped = self.W;
           for i = 1:self.nlayers
               W_dropped{i}(dropped{i}, :) = 0; % Dropped nodes have 0 output.
               % By scaling W up during dropout training, we don't have to 
               % scale it down during test. Simplifies code. 
               W_dropped{i} = W_dropped{i} / (1 - self.dropprob(i));
           end
        end
        function [E dE_dW] = gradient_dropout(self, data)
            % Dropout nodes and compute gradient.
            dropped = sampledrop(self);
            % Below I perform a shortcut for computing the dropout grad.
            % 1) Compute activation by setting output weights to zero.
            % 2) Standard gradient holds for most weights, except I need to
            % delete the gradient of the weights coming out of dropped
            % nodes. Note: Lower layer weights have correct gradients
            % because gradient contribution through dropped nodes is 0.
            W_dropped = dropoutweights(self, dropped);
            [E, dE_dW] = feval(self.GradFcn, W_dropped, data.input, data.target);
            % Kill gradient of output weights of dropped nodes to zero.
            for k = 1:self.nlayers
                dE_dW{k}(dropped{k}, :) = 0;
            end 
        end 
    end
end