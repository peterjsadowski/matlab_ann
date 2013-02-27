classdef mtmlp < mlp
% General mlp with multitask output:
% Task 1: Classification (Single or multi class.)
% Task 2: Autoencode [0,1] input.
% This code should act the same as a standard NN w/ backprop (see netbp.m),
% when alpha=0.
    properties
        nepoch    = 10;       % Number of epochs.
        batchsize = 1000;     % Number of examples in each batch.
        nupdate  = 3;         % Number of linesearches/updates per batch.
        alpha    = 0;         % Alpha parameter, [0,1]: proportion of error coming from autoencoding task.
        autoencoder;          % Perceptron object for last layer autoencoding.
        debug    = false;     % Prints objective function.
        limitbackprop = false;% If true, limit backprop to top two layers.
        limitbackprop1= false;% If true, limit backprop to top one layer.
        convflag      = false;% If true, stop when convergence criteria are met.
        convthreshold = 1e-5; % 
        convwindow    = 1;  % Convergence window.
        convmin       = 10;   % Minimum number of iterations before convergence possible.
    end
    
    properties (Dependent = true)
        GradFcn
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        % Initialization
        function self = mtmlp(varargin)
            % Initialize first part of the network just as we do for netbp.
            [fullarch, TransferFcns, ErrorFcn] = mtmlp.processArgs(varargin{:});
            assert(ischar(ErrorFcn));
            self = self@mlp(fullarch, TransferFcns);
            self.ErrorFcn = ErrorFcn;
            self = initialize(self); % Initialize weights.
            % Initialize autoencoder top layer. This is parallel to and separate from the last layer.
            self.autoencoder = perceptron(self.layers{end}.NumInputs,...
                                          self.layers{1}.NumInputs,...
                                          'logsig', ...
                                          'CrossEntropyError');
            self.autoencoder.initscale = self.initscale;
            self.autoencoder = initialize(self.autoencoder);
        end
        function GradFcn = get.GradFcn(self)
            % Return gradient function (with alpha specified).
            % For the moment we are assuming the labeled data is 
            GradFcn = @(W, data)(gradient_mt(W, data, self.alpha));
        end
        function self = set.GradFcn(self, fcn) %#ok<INUSD>
            % Gradient function is simply gradient_crossent_mt with alpha
            % specified. DO NOTHING.
            error('The GradFcn is set automatically.')
            return %#ok<UNRCH>
        end
        
        % Computations
        function self = train(self, data)
            % Train network.
            
            % Prepare data.
            if ischar(data), self.dataset=data; data = loaddata(data); end;
            assert(min(data.input(:)) >= 0 && max(data.input(:)) <= 1); % Our autoencoding assumes this.
            if size(data.target, 2) ~= self.NumOutputs
                error('Data and architecture size do not agree.')
            end
            batchdata = makeBatchData(data, self.batchsize);
            nbatch = length(batchdata);
            
            % Record initial error.
            if isempty(self.ErrorTrain)
                self = recordErrors(self, data);
                printstatus(self);
            end
            
            % Iterate
            for i = 1:self.nepoch
                for j = 1:nbatch
                    self = updateWeights(self, batchdata{j});
                end % batch
                
                % Record train, test error rates on whole dataset.
                self = recordErrors(self, data);
                % Print to screen.
                printstatus(self);
                % Detect convergence for multitask.
                if self.convflag && hasConverged(self.ErrorTrain,...
                                                 'minsteps', self.convmin, ...
                                                 'window', self.convwindow,...
                                                 'threshold', self.convthreshold)
                     % Stop training.
                     fprintf('Convergence criteria reached.\n')
                     break
                end
                
            end % epoch
        end
        function self = recordErrors(self, data)
            % Updates ErrorTrain and ErrorTest using data.input, data.testinput.
            % Also calculate classification error (for classification task)
            
            % Evaluate error function on all data.
            MTW = [self.W; self.autoencoder.W]; % Multitask weights.
            [Etrain, ~] = (feval(self.GradFcn, MTW, data));
            testdata.input = data.testinput;
            testdata.target = data.testtarget;
            [Etest, ~] = (feval(self.GradFcn, MTW, testdata));
            self.ErrorTrain(end+1) = Etrain;
            self.ErrorTest(end+1)  = Etest;
            
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
        
        % Output
        function string = param2str(self)
            % Creates short string describing parameters of netdt.
            if self.convflag
                time = clock;
                string = sprintf('mtmlp_%s_bs%d_nup%d_convthresh%d_day%dhr%dmin%dsec%d',...
                    arch2str(self), self.batchsize, self.nupdate, -log10(self.convthreshold), time(3), time(4), time(5), floor(time(6)));
            else
                string = sprintf('mtmlp_%s_bs%d_nup%d', arch2str(self), self.batchsize, self.nupdate);
            end
        end
        
    end
    
    % ==================================================
    % PROTECTED METHODS
    % ==================================================
    methods (Access='protected')
        function self = updateWeights(self, data)
            % Perform a single weight update on multitask weight cell array MTW and data struct. 
            MTW = [self.W; self.autoencoder.W]; % Multitask weights.
            
            % Special case for limitbackprop flag:
            if self.limitbackprop && (self.nlayers > 2)
                % Only update the last three weight matrices.
                % To do this, optimize the last two layers (3 weight matrices)
                % on the input to that layer.
                if self.debug
                    fprintf('Updating layers %d and %d.\n', self.nlayers-1, self.nlayers)
                end
                % Only update top 3 weight matrices.
                MTW3 = MTW(end-2:end);
                % Target of encoder is still input.
                data.target2 = data.input;
                % Input is layerk activation.
                data.input     = applyk1k2(self, 1, (self.nlayers-2), data.input);
                % Unlabeled input is layerk activation.
                if isfield(data,'unlabeled')
                    data.targetunlabeled = data.unlabeled;
                    data.unlabeled = applyk1k2(self, 1, (self.nlayers-2), data.unlabeled);
                end
                % 
                [MTW3, E, nsteps] = minimize(MTW3,self.GradFcn,self.nupdate,true,data);
                MTW(end-2:end) = MTW3;
            elseif self.limitbackprop1 && (self.nlayers > 1)
                % Only update top 3 weight matrices.
                MTW2 = MTW(end-1:end);
                % Target of encoder is still layer1 input.
                data.target2 = data.input; 
                % Input is layerk activation.
                data.input = applyk1k2(self, 1, (self.nlayers-1), data.input);
                % Unlabeled input is layerk activation.
                if isfield(data,'unlabeled')
                    data.targetunlabeled = data.unlabeled;
                    data.unlabeled = applyk1k2(self, 1, (self.nlayers-1), data.unlabeled);
                end
                [MTW2, E, nsteps] = minimize(MTW2,self.GradFcn,self.nupdate,true,data);
                MTW(end-1:end) = MTW2;
            else
                % Standard method: backprop all the way.
                [MTW, E, nsteps] = minimize(MTW, self.GradFcn, self.nupdate, true, data);

            end
            % Update weights.
            self.W = MTW(1:end-1);
            self.autoencoder.W = MTW(end);
            
            if self.debug
                fprintf('Objective: %f,\tnsteps: %d\n' , E, nsteps);
            end
        end
        
    end
    
    % ==================================================
    % STATIC METHODS
    % ==================================================
    methods (Static = true, Access='protected')
        function [fullarch, TransferFcns, ErrorFcn] = processArgs(varargin)
            % Process various forms of input for constructor.
            % INPUT: (arch, data)
            %     OR (fullarch, TransferFcns, ErrorFcn)
            % OUTPUT:
            %   fullarch = vector specifying n0,...,nl
            %   TransferFcns = cellarray specifying all transfer functions
            %   ErrorFcn =
            if nargin == 2
                % Convienient constructor from hidden arch and data.
                arch = varargin{1};
                data = varargin{2};
                if ischar(data), data = loaddata(data); end;
                fullarch = [size(data.input,2), arch, size(data.target,2)];
                [TransferFcns, ErrorFcn] = mlp.detectFcns(fullarch, data);
            elseif nargin == 3
                % Fully specified.
                fullarch = varargin{1};
                TransferFcns = varargin{2};
                ErrorFcn = varargin{3};
            else
                error('Input error.')
            end
            if ~iscell(TransferFcns)
                % Assign this transfer function to every layer.
                assert(ischar(TransferFcns));
                TransferFcns = repmat({TransferFcns}, 1, length(fullarch)-1);
            end
            if iscell(ErrorFcn) && length(ErrorFcn)==1
                ErrorFcn = ErrorFcn{1}; % Common mistake.
            end
        end
    end
end

    
