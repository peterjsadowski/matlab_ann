classdef perceptron_logsig_crossent < perceptron
    % Layer of standard neural network.
    properties
    
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods (Abstract = true)
        function train(self, data)
            input = data.input;
            nepoch = 1;
            nu = 0.1;
            stochasticGradientDescent(self, input, nepoch, nu)
        end
        
        function stochasticGradientDescent(self, input, target, nepoch, nu)
            % Simple stochastic gradient descent.
            m = size(input,1);
            for i = 1:nepoch
                for j = 1:m
                    self.W = self.W - nu * calcGradient(self, input(j,:), target(j,:));
                end
            end
        end
        
        function calcGradient(self, input, target)
            m = size(input, 1);
            Y = apply(self, input);
            switch self.TransferFcn
                case 'logsig'
                    switch self.ErrorFcn
                        (Y - target)
                case 'threshold'
                    
                otherwise
                    error('Unknown transfer function. Try: logsig, threshold,...')
                    
        end
        
        
                            
                % For 0,1 setup, dE/dwkj ~ (bk - data)*bj in perceptron learning algorithm.
X1 = [input, ones(m,1)];
X2 = (X1 * W) > 0; % mxn. Bias term added.
dE_dx2   = X2 - target;         % mxn2
dx2_dw  = [input, ones(m,1)];   % mxn1 Not actually, but approx.
dE{1}   = X1' * dE_dx2;            % n1xn2

% Hamming distance.
E = biterr(target, X2); 


                [E dE_dW] = feval(funcname, net2weightcell(net.layers{k}.W), layerdata.input, layerdata.target);
                dE_dW = dE_dW{1}; % Single layer.
                dE_dW = dE_dW / size(layerdata.input, 1); % Scale so that dE is average per sample.
        
    end