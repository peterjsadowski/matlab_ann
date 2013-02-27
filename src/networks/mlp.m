classdef mlp
    % Multilayer perceptron.
    % Usage:
    % m = mlp(fullarch, TransferFcn(s));
    properties
        layers
        initscale = 1   % Initial scale of weights. (Var of sum)
        dataset = []
        
        ErrorTrain = [] % Array with error after each epoch.
        ErrorTest = []  % Array with error after each epoch.
        ErrorTrainClass = [] % Classification training error after each epoch.
        ErrorTestClass = []  % Classification test error after each epoch.
    end
    
    % ==================================================
    % DEPENDENT PROPERTIES
    % ==================================================
    properties (Dependent = true) %(Constant) (GetAccess=private)
        W
        fullarch
        nlayers % Number of layers.
        NumInputs
        NumOutputs
        TransferFcns
        ErrorFcn  % Error function of output layer.
    end
    properties (Access = protected)
        ErrorFcn_LastLayer = [];
    end
    % ==================================================
    % ABSTRACT METHODS
    % ==================================================
%     methods (Abstract = true)
%         self = train(self, data)
%     end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        % Initialization
        function self = mlp(fullarch, TransferFcns)
            if ~iscell(TransferFcns)
               % Assign this transfer function to every layer.
               assert(ischar(TransferFcns));
               temp = TransferFcns;
               clear TransferFcns
               for i = 1:length(fullarch)-1
                   TransferFcns{i} = temp;
               end
            end
            % Create layers.
            for i = 1:length(fullarch)-1
                self.layers{i} = perceptron(fullarch(i), fullarch(i+1), TransferFcns{i});
            end
            self = initialize(self);
        end
        function self = initialize(self)
            % Initialize weights of each layer so that sum has var initscale.
            for i = 1:self.nlayers
                self.layers{i}.initscale = self.initscale;  % Set initscale.
                self.layers{i} = initialize(self.layers{i});% Random weights.
            end
            % Reset Errors
            self.ErrorTrain = [];
            self.ErrorTest = [];
            self.ErrorTrainClass = [];
            self.ErrorTestClass = [];
        end
        function self = initializeBias(self, bias)
            % Set bias of all nodes. By default this is 0.
            for i = 1:self.nlayers
                self.layers{i} = initializeBias(self.layers{i}, bias);
            end
        end
        function self = pretrain_rbm(self, data, niter)
            % Pretrain weights as a stack of sequential RBMs.
            if isstruct(data)
                X = data.input;
            else
                X = data;
            end
            for i = 1:self.nlayers
               self.layers{i} = self.layers{i}.pretrain_rbm(X, niter);
               X = apply(self.layers{i}, X);
            end
        end
        % Properties
        function NumInputs=get.NumInputs(self)
            % Number of inputs not including bias term.
            NumInputs = self.layers{1}.NumInputs;
        end
        function NumOutputs=get.NumOutputs(self)
            NumOutputs = self.layers{end}.NumOutputs;
        end
        function nlayers=get.nlayers(self)
            nlayers = length(self.layers);
        end
        function fullarch=get.fullarch(self)
            fullarch = self.layers{1}.NumInputs;
            for i = 1:self.nlayers
                fullarch = [fullarch, self.layers{i}.NumOutputs];
            end
        end
        function TransferFcns=get.TransferFcns(self)
            TransferFcns = cell(self.nlayers,1);
            for i = 1:self.nlayers
                TransferFcns{i} = self.layers{i}.TransferFcn;
            end
        end
        function self=set.TransferFcns(self, TransferFcns)
            % Set the TransferFcn property of each layer.
            if ~iscell(TransferFcns)
                % Assign this transfer function to every layer.
                assert(ischar(TransferFcns));
                TransferFcns = repmat({TransferFcns}, 1, self.nlayers);
            end
            for i = 1:self.nlayers
                self.layers{i}.TransferFcn = TransferFcns{i};
            end
        end
        function ErrorFcn=get.ErrorFcn(self)
            ErrorFcn = self.ErrorFcn_LastLayer;
        end
        function self=set.ErrorFcn(self, ErrorFcn)
            self.ErrorFcn_LastLayer = ErrorFcn;
        end
        function W = get.W(self)
            % Return cell array of weights.
            W = cell(self.nlayers, 1);
            for i = 1:self.nlayers
                W{i} = self.layers{i}.W;
            end
        end
        function self = set.W(self, W)
            % Sets each layer W from cell array.
           for i = 1:self.nlayers
               self.layers{i}.W = W{i};
           end
        end 
        % Computations
        function output = apply(self, input)
            % Compute output of network from input.
            output = input;
            for i = 1:self.nlayers
                output = apply(self.layers{i}, output);
            end
        end
        function output = applyk(self, k, activationk)
            % Compute output of network conditioned on given activation at layer k.
            output = activationk;
            for i = (k+1):self.nlayers
                output = apply(self.layers{i}, output);
            end
        end
        function output = applyk1k2(self, k1, k2, k1input)
            % Compute output of k2 conditioned on input to k1.
            output = k1input;
            for i = k1:k2
                output = apply(self.layers{i}, output);
            end
        end
        function self = recordErrors(self, data)
            % Updates ErrorTrain and ErrorTest using data.input, data.testinput.
            error = mean(feval(self.ErrorFcn, apply(self, data.input), data.target));
            self.ErrorTrain = [self.ErrorTrain, error];
            if isfield(data, 'testtarget')
                % Record ErrorTest
                error = mean(feval(self.ErrorFcn, apply(self, data.testinput), data.testtarget));
                self.ErrorTest  = [self.ErrorTest, error];
            end
            % Update ErrorTrainClass and ErrorTestClass if appropriate.
            switch self.ErrorFcn
                case 'MulticlassCrossEntropyError'
                    % Record ErrorTrainClass as mean ZeroOne error.
                    error = mean(feval('MulticlassZeroOneError', apply(self, data.input), data.target));
                    self.ErrorTrainClass  = [self.ErrorTrainClass, error];
                    if isfield(data, 'testtarget')
                        % Record ErrorTestClass
                        error = mean(feval('MulticlassZeroOneError', apply(self, data.testinput), data.testtarget));
                        self.ErrorTestClass  = [self.ErrorTestClass, error];
                    end
                case {'CrossEntropyError', 'Hamming'}
                    % Record ErrorTrainClass as mean bit-wise error.
                    % If values are real, then use x >= 0.5.
                    error = mean(feval('MeanBitwiseError', apply(self, data.input), data.target));
                    self.ErrorTrainClass  = [self.ErrorTrainClass, error];
                    if isfield(data, 'testtarget')
                        % Record ErrorTestClass
                        error = mean(feval('MeanBitwiseError', apply(self, data.testinput), data.testtarget));
                        self.ErrorTestClass  = [self.ErrorTestClass, error];
                    end
            end
        end
        function X = getActivations(self, data)
            % Returns activations at each layer for input data.
            if ischar(data), data = loaddata(data); data = data.input; end;
            X = cell(1, self.nlayers);
            X{1} = apply(self.layers{1}, data);
            for i = 2:self.nlayers
                X{i} = apply(self.layers{i}, X{i-1});
            end
        end       
        function X = getActivationsk1k2(self, data, k1, k2)
            % Return activations of k1-k2 conditioned on input to k1.
            assert(k2 >= k1);
            layeridx = k1:k2;
            nlayers = length(layeridx);
            X = cell(1, nlayers);
            X{1} = apply(self.layers{layeridx(1)}, data);
            for i = 2:nlayers
                X{i} = apply(self.layers{layeridx(i)}, X{i-1});
            end
        end
        function X = getStochasticActivations(self, data)
            % Returns activations at each layer for input data.
            if ischar(data), data = loaddata(data); data = data.input; end;
            X = cell(1, self.nlayers);
            X{1} = apply_sigmoid_stochastic(self.layers{1}, data);
            for i = 2:self.nlayers
                X{i} = apply_sigmoid_stochastic(self.layers{i}, X{i-1});
            end
        end
        % Error Plots
        function printstatus(self)
            % Simply print error to screen.
            i = length(self.ErrorTrain) - 1; % First is initialization.             
            if ~isempty(self.ErrorTestClass)
                % Train and test class error.
                fprintf('Iteration:%d\tErrorTrain:%0.06f\tErrorTest:%0.06f\tErrorClass:%0.06f\tErrorClassTest:%0.06f\n', ...
                    i, self.ErrorTrain(end), self.ErrorTest(end), self.ErrorTrainClass(end), self.ErrorTestClass(end));
            elseif ~isempty(self.ErrorTrainClass)
                % Train and test class error.
                fprintf('Iteration:%d\tErrorTrain:%0.15f\tErrorTrainClass:%0.06f\n', i, self.ErrorTrain(end), self.ErrorTrainClass(end));
            elseif ~isempty(self.ErrorTest)
                % Train and test error.
                fprintf('Iteration:%d\tErrorTrain:%0.15f\tErrorTest:%0.06f\n', i, self.ErrorTrain(end), self.ErrorTest(end));
            else
                % Just train error.
                fprintf('Iteration:%d\tErrorTrain:%0.15f\n', i, self.ErrorTrain(end));
            end
        end            
        function plotError(self)
            % Plot error trajectory for train and test data ErrorTrain, ErrorTest. 
            figure(1); clf;
            hold on            
            plot(self.ErrorTrain, 'k-');
            plot(self.ErrorTest, 'b-');
            hold off
            xlabel('Epoch')
            ylabel('Error')
            names = {'Train', 'Test'};
            legend(names)
        end
        function plotErrorClass(self)
            % Plot error trajectory for train and test data ErrorTrain, ErrorTest. 
            figure(1); clf;
            hold on            
            plot(self.ErrorTrainClass, 'k-');
            plot(self.ErrorTestClass, 'b-');
            hold off
            xlabel('Epoch')
            ylabel('Classification Error')
            names = {'Train', 'Test'};
            legend(names)
        end
        function auc = calcAUC(self, input, target)
            % Compute Area Under ROC Curve
            pred = self.apply(input);
            [tpr, fpr, thresholds] = roc(target', pred');
            auc = self.areaundercurve(fpr, tpr);
        end
        function plotROC(self, input, target)
            % Plot ROC curve.
            pred = self.apply(input);
            plotroc(target', pred');
        end
        % Saving
        function str = arch2str(self)
            % Return short string describing arch of mlp.
            arch = self.fullarch;
            if all(arch == arch(1))
                str = sprintf('%dx%d', arch(1), length(arch));
            elseif length(arch) < 10
                % Show all x in the form n0_n1_n2_..._nx
                str = sprintf('_%d', arch);
                str = str(2:end); % Remove leading underscore.
            else
                % Show first x then _etc.
                str = sprintf('_%d', arch(1:10));
                str = [str(2:end), '_etc']; % Remove leading underscore
            end
        end
        function string = param2str(self)
            % Creates short string describing parameters of mlp.
            string = sprintf('mlp_%s_iw%0.2f', arch2str(self), self.initscale);
        end
        function savenet(self, resultsdir)
            % Save network to mat file.
            if nargin < 2
                resultsdir = 'results';
            end
            % Make sure resultsdir exists.
            [~,~,~] = mkdir(resultsdir);
            n = self;
            save([resultsdir, '/', param2str(self), '.mat'], 'n');
        end
        
        

    end
    % ==================================================
    % PRIVATE METHODS
    % ==================================================
    methods (Access='private')
        
    end   
    % ==================================================
    % STATIC METHODS
    % ==================================================
    methods (Static = true, Access='protected')
        function [TransferFcns, ErrorFcn] = detectFcns(fullarch, data)
            % Detect default TransferFcns and ErrorFcn from fullarch and data.
            nlayers = length(fullarch) - 1;
            if all(1 == sum(data.target, 2)) && all(1 == max(data.target, [], 2))
                % Multiclass classification with SoftmaxTransfer output.
                TransferFcns = repmat({'logsig'}, 1, nlayers);
                TransferFcns{end} = 'SoftmaxTransfer';
                ErrorFcn = 'MulticlassCrossEntropyError';
            elseif all((data.target(:) >= 0)) && all((data.target(:) <= 1))
                % Probabilistic output. Cross-ent error.
                TransferFcns = repmat({'logsig'}, 1, nlayers);
                ErrorFcn = 'CrossEntropyError';
            else
                % Real valued output, sum of squares error.
                TransferFcns = repmat({'logsig'}, 1, nlayers);
                TransferFcns{end} = 'linear';
                ErrorFcn = 'SumSquaredError';
            end
        end
        function auc = areaundercurve(FPR,TPR)
            % Given true positive rate (TPR) and false positive rate (FPR)
            % calculates the area under the curve (AUC).
            % True positive are on the y-axis and false positives on 
            % the x-axis; sum rectangular area between all points.
            [x2,inds] = sort(FPR);
            x2 = [x2,1];      % Trick: invent a point 1,1
            y2 = TPR(inds);
            y2 = [y2,1];
            xdiff = diff(x2);
            xdiff = [x2(1),xdiff];
            auc1 = sum(y2.*xdiff); % upper point area
            auc2 = sum([0,y2([1:end-1])].*xdiff); % lower point area
            auc = mean([auc1,auc2]);
        end
    end
end

% ==================================================
% OTHER METHODS
% ==================================================


