classdef netbp < mlp
% General neural net w/ differentiable transfer trained using backpropagation.
    properties
        nepoch    = 10;      % Number of epochs.
        batchsize = 1000;    % Number of examples in each batch.
        nupdate  = 3;        % Number of linesearches/updates per batch.
        GradFcn  = [];       % Gradient function - depends on transfer and error functions.
        convflag = false;    % If true, stop when convergence criteria are met.
        convthreshold = 1e-5;% Convergence if we see improvement less than this.
        convwindow    = 1;   % Convergence window (must see improvement over window iters).
        convmin       = 10;  % Minimum number of iterations before convergence possible.
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        function self = netbp(varargin)
            % Usage: netbp(arch, data) OR netbp(fullarch, TransferFcns, ErrorFcn)
            [fullarch, TransferFcns, ErrorFcn] = netbp.processArgs(varargin{:});
            assert(ischar(ErrorFcn));
            self = self@mlp(fullarch, TransferFcns);
            self.ErrorFcn = ErrorFcn;
            self.GradFcn = getGradientFcn(TransferFcns, ErrorFcn);
            self = initialize(self); % Initialize weights.
        end
        
        function self = train(self, data)
            % Train network.
            if ischar(data), self.dataset=data; data = loaddata(data); end;
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
                    [self.W, ~, ~] = minimize(self.W, ...
                                              self.GradFcn, ... # Gradient
                                              self.nupdate, ... # Num linesearches
                                              true, ...
                                              batchdata{j}.input, ...
                                              batchdata{j}.target);
                end % batch
                
                % Record train, test error rates.
                self = recordErrors(self, data);
                % Print to screen.
                printstatus(self);
                % Detect convergence.
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
        
        % Output
        function string = param2str(self)
            % Creates short string describing parameters of netdt.
            string = sprintf('bp_%s_bs%d_nup%d', arch2str(self), self.batchsize, self.nupdate);
        end

    end
    % ==================================================
    % PRIVATE METHODS
    % ==================================================
    methods (Access='private')
        function self = train_batchgd_logsig_crossent(self, data, nepoch, nu, batchsize)
            % Gradient descent in batchmode for logsig transfer and crossent error.
            batchdata = makeBatchData(data, batchsize);
            for i = 1:nepoch
                for j = 1:length(batchdata)
                    dE_dW = addbias(batchdata{j}.input)' * (apply(self, batchdata{j}.input) - batchdata{j}.target);
                    self.W = self.W - nu * dE_dW;
                end
            end
        end
        function self = train_batchgd_theshold_hamming(self, data, nepoch, nu, batchsize)
            % Gradient descent in batchmode for threshold transfer and Hamming error.
            % (Perceptron learning algorithm)
            batchdata = makeBatchData(data, batchsize);
            for i = 1:nepoch
                for j = 1:length(batchdata)
                    dE_dW = addbias(batchdata{j}.input)' * (apply(self, batchdata{j}.input) - batchdata{j}.target);
                    self.W = self.W - nu * dE_dW;
                end
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
% ==================================================
% OTHER METHODS
% ==================================================

    
