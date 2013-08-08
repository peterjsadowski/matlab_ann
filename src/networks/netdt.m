classdef netdt < mlp
    % General neural net trained using deep target algorithms.
    % ==================================================
    % PROPERTIES
    % ==================================================
    properties
        savefreq = 0;         % Save network every few epochs.
        savedir  = 'results'; % Directory tosave results to. 
        
        nepoch    = 10;       % Number of epochs.
        batchsize = 1000;     % Number of examples in each batch.
        
        % 1 is good. Otherwise, weights will change then move back. 
        nupdate  = 1;        % Max number of linesearches/updates per batch.
        
        
        nsamp = 1000;  % Max number of binary vectors to try as targets.
        nvarnodes = inf; % Number of variable nodes when updating targets. (for opttargetfcn=optimizeTargets)
        
        schedule = [];
        layertrainfcn = 'train_gd'; % Method of class perceptron.
        opttargetfcn = 'optimizeTargets_local'; % Method of class netdt.
    end
    properties (Dependent = true)
        nu        % Learning rate.
        mu        % L2 penalty on weights, weight decay
        momentum  % Momentum parameter.
        ErrorFcns % Error function of each layer (used for training).
        ErrorFcn_Targets % Possibly use different (faster) error function for choosing targets.
        % Analysis
    end
    properties (Access = 'private')
        learning_rate  % nu
        L2_penalty     % mu
        momentum_param % momentum
        error_fcn_targets = [] % Possibly use different (faster) error function for choosing targets.
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        % Most Common
        function self = netdt(varargin)
            % Usage: netbp(arch, data) OR netbp(fullarch, TransferFcns, ErrorFcns)
            [fullarch, TransferFcns, ErrorFcns] = netdt.processArgs(varargin{:});
            % Instantiate
            self = self@mlp(fullarch, TransferFcns);
            self.ErrorFcn = ErrorFcns{end}; % Should usually be the same.
            self.ErrorFcn_Targets = self.ErrorFcn;
            self.ErrorFcns = ErrorFcns;     % Error functions for each layer.
            % Default parameters
            self.initscale = 4*sqrt(3);
            self.nu = 1;
            self.momentum = 0;
            self.schedule = 1:self.nlayers;
            % Initialize weights
            self = initialize(self);        % Initialize weights.
        end
        function self = train(self, data)
            % Train network.
            % Get batchdata.
            if ischar(data), self.dataset=data; data = loaddata(data); end;
            batchdata = makeBatchData(data, self.batchsize);
            nbatch = length(batchdata);
            % Record initial error.
            if isempty(self.ErrorTrain)
                self = recordErrors(self, data);
                printstatus(self);
            end
            % Begin learning
            X = cell(1, self.nlayers);% Initial activations.
            for i = 1:self.nepoch
                for j = 1:nbatch
                    if length(self.schedule) ~= self.nlayers
                        % Need to get activations.
                        X = getActivations(self, batchdata{j}.input);
                    end
                    for k = self.schedule
                        % Find layerdata.input.
                        if k==1
                            layerdata.input = batchdata{j}.input;
                        else
                            layerdata.input = X{k-1}; % Should be updates.
                        end
                        
                        % Find layerdata.target.
                        initialk = apply(self.layers{k}, layerdata.input);
                        layerdata.target = feval(self.opttargetfcn, self, k, initialk, batchdata{j}.target);
                        %layerdata.target = optimizeTargets(self, k, initialk, batchdata{j}.target);

                        % Update.
                        %self.layers{k} = train_gd_adapt(self.layers{k}, layerdata, self.nupdate);
                        %temp = self.layers{k};
                        self.layers{k} = feval(self.layertrainfcn, self.layers{k}, layerdata, self.nupdate);
                        
                        % Debug
%                         self = recordErrors(self, data);
%                         fprintf('L%d ', k); printstatus(self)
%                         if i>3 && self.ErrorTrain(end) > self.ErrorTrain(end-1)
%                             a = initialk;
%                             b = layerdata.target;
%                             c = apply(self.layers{k}, layerdata.input);
%                             Xend_initial = applyk(self, k, initialk);
%                             Xend_target = applyk(self, k, layerdata.target);
%                             Xend_new = applyk(self, k, c);
%                             E_initial = mean(feval(self.ErrorFcn, Xend_initial, batchdata{j}.target));
%                             E_target = mean(feval(self.ErrorFcn, Xend_target, batchdata{j}.target));
%                             E_new = mean(feval(self.ErrorFcn, Xend_new, batchdata{j}.target));
%                             E_loc_init = mean(feval(self.layers{k}.ErrorFcn, a, layerdata.target));
%                             E_loc_new = mean(feval(self.layers{k}.ErrorFcn, c, layerdata.target));
%                             keyboard
%                         end
                        
                        % Compute output of this layer. Used for next layer
                        X{k} = apply(self.layers{k}, layerdata.input);                        
                    end %layer
                end % batch
                % Record errors
                self = recordErrors(self, data);
                printstatus(self)
                % Save at intervals, and at final epoch.
                if (mod(i,self.savefreq) == 0) || (i==self.nepoch)
                    self.savenet(self.savedir)
                end
                %self.zeroNodes(:, end+1) = countZeroNodes(self, data);
                %fprintf('%d ', self.zeroNodes(:,end));
                %fprintf('\n');
            end % epoch
        end
        function self = convert2threshold(self)
           self.TransferFcns = 'threshold';
           self.ErrorFcn = 'Hamming';
           self.ErrorFcn_Targets = 'Hamming';
           self.ErrorFcns = 'Hamming';
        end
        % Properties
        function ErrorFcn_Targets = get.ErrorFcn_Targets(self)
            % If no particular function is specified, use the network
            % error function.
            if ~isempty(self.error_fcn_targets)
                ErrorFcn_Targets = self.error_fcn_targets;
            else    
                ErrorFcn_Targets = self.ErrorFcn;
            end
        end
        function self = set.ErrorFcn_Targets(self, ErrorFcn_Targets)
           self.error_fcn_targets = ErrorFcn_Targets;
        end
        function ErrorFcns = get.ErrorFcns(self)
            % Get ErrorFcn property of each layer perceptron.
            ErrorFcns = cell(self.nlayers,1);
            for i = 1:self.nlayers
                ErrorFcns{i} = self.layers{i}.ErrorFcn;
            end
        end
        function self = set.ErrorFcns(self, ErrorFcns)
            % Set ErrorFcn property of each layer perceptron.
            if ~iscell(ErrorFcns)
                % Assign this error function to all layers.
                assert(ischar(ErrorFcns));
                ErrorFcns = repmat({ErrorFcns}, 1, self.nlayers);
            end
            for i = 1:self.nlayers
               self.layers{i}.ErrorFcn = ErrorFcns{i};
            end
            % Check
            if ~strcmp(self.layers{end}.ErrorFcn, self.ErrorFcn)
                warning('Training ErrorFcns{end} ~= ErrorFcn. Make sure you know what you are doing.');
            end
        end
        function nu = get.nu(self)
            % Returns high level nu.
            nu = self.learning_rate;
        end
        function self = set.nu(self, nu)
            % Set adjusted nu for all layers; keep track of nu in this
            % object too.
            self.learning_rate = nu;
            % Set parameters layer. Scale nu by number of inputs.
            for k = 1:self.nlayers
                %self.layers{k}.nu = self.layers{k}.calcAdjustednu(self.nu);
                self.layers{k}.nu = nu;
            end
        end
        function nu = get.mu(self)
            % Returns high level mu.
            nu = self.L2_penalty;
        end
        function self = set.mu(self, mu)
            % Set adjusted mu for all layers; keep track of mu in this
            % object too.
            self.L2_penalty = mu;
            % Set parameters layer. Scale nu by number of inputs.
            for k = 1:self.nlayers
                %self.layers{k}.nu = self.layers{k}.calcAdjustednu(self.nu);
                self.layers{k}.mu = mu;
            end
        end
        function momentum = get.momentum(self)
            % Return high-level momentum.
            momentum = self.momentum_param;
        end
        function self = set.momentum(self, momentum)
            % Set momentum term in each layer.
            self.momentum_param = momentum;
            for k = 1:self.nlayers
                %self.layers{k}.nu = self.layers{k}.calcAdjustednu(self.nu);
                self.layers{k}.momentum = momentum;
            end
        end
        % Computations
        function targetk = optimizeTargets_limited(self, k, initialk, target)
            % Optimize layer k targets, sampling from nvarnodes at a time. 
            % 
            % k = layer idx -- we need target for this layer.
            % initialk = m x n_(k) matrix of layer k activation hk.
            % target = target for layer l.
            if k == self.nlayers
                % We already have the target h^l.
                targetk = target;
                return
            end
            % Add noise to nvarnodes features at a time.
            nvariable = min(self.nvarnodes, size(initialk,2));
            flipbitidx = randperm(size(initialk, 2), nvariable);
            
            m = size(target, 1);
            targetk = zeros(size(initialk));
            for i = 1:m
                % Find sample set S_ik.
                if 2^nvariable <= self.nsamp
                    % Do all possible.
                    S_ik = repmat(initialk(i,:), 2^nvariable + 1, 1);
                    newbits = de2bi(0:(2^nvariable - 1));
                    S_ik(2:end, flipbitidx) = newbits;
                else
                    S_ik = repmat(initialk(i,:), self.nsamp + 1, 1);
                    S_ik(2:end, flipbitidx) = rand(self.nsamp, nvariable) < 0.5;
                end

                % Select target from S_ik
                targetk(i, :) = selecttargetk_single(self, k, S_ik, target(i,:));
            end
        end
        function targetk = optimizeTargets_local(self, k, initialk, target)
            % Optimize layer k targets, sampling locally.
            % Parameterization: nvarnodes = E[bitflips], nsamp 
            % k        = layer idx -- we need target for this layer.
            % initialk = m x n_(k) matrix of layer k activation hk.
            % target   = target for layer l.
            binarytargets = (length(unique(initialk(:))) <= 2);
            %assert(binarytargets, 'optimizeTargets_local doesnt make sense for nonbinary targes'); 
            
            if k == self.nlayers
                targetk = target;  % We already have the target h^l.
                return
            end
            [m, n] = size(initialk);
            targetk = zeros(size(initialk));
            for i = 1:m
                % Find sample set S_ik.
                
                S_ik = repmat(initialk(i,:), self.nsamp, 1);
                %noiseidx = randi(numel(S_ik), 1, 2 * size(initialk,1));
                %S_ik(noiseidx) = not(S_ik(noiseidx));
                
                noise = rand(self.nsamp, n) < (self.nvarnodes / n); % E[nonzero]=nvarnodes
                noise(1,:) = 0; % Keep original activation in row 1.
                if binarytargets
                    % Flip bits
                    S_ik(noise) = not(S_ik(noise));
                else
                    % Round to binary, then flip.
                    S_ik(noise) = not(S_ik(noise) > 0.5);
                end
                
                % Select target from S_ik
                targetk(i, :) = selecttargetk_single(self, k, S_ik, target(i,:));
            end
        end
        function targetk = optimizeTargets_sparse(self, k, initialk, target)
            % Optimize layer k targets, with sparse sample set.
            % k = layer idx -- we need target for this layer.
            % initialk = m x n_(k) matrix of layer k activation hk.
            % target = target for layer l.
            if k == self.nlayers
                % We already have the target h^l.
                targetk = target;
                return
            end
            [m, n] = size(initialk);
            targetk = zeros(size(initialk));
            for i = 1:m
                % Find sample set S_ik.
                
                %S_ik = repmat(initialk(i,:), self.nsamp, 1);
                %noiseidx = randi(numel(S_ik), 1, 2 * size(initialk,1));
                %S_ik(noiseidx) = not(S_ik(noiseidx));
                
                % Sparse binary targets.
                S_ik = rand(self.nsamp, n) < (self.nvarnodes / n); % E[nonzero]=nvarnodes
                S_ik(1,:) = initialk(i,:); % Keep original activation.
                % Select target from S_ik
                targetk(i, :) = selecttargetk_single(self, k, S_ik, target(i,:));
            end
        end
        function targetk = optimizeTargets_uniform(self, k, initialk, target)
            % Optimize layer k targets with uniform samples.
            % k = layer idx -- we need target for this layer.
            % initialk = m x n_(k) matrix of layer k activation hk.
            % target = target for layer l.
            if k == self.nlayers
                % We already have the target h^l.
                targetk = target;
                return
            end
            % Find sample set S.
            m = size(target, 1);
            if self.layers{k}.NumOutputs <= 10
                % Sample all binary targets. 
                S = de2bi(0:(2^self.layers{k}.NumOutputs - 1));
                % If real valued output, include initial activation too
                if strcmp(self.layers{k}.TransferFcn, 'threshold')
                    hkidx = bi2de(initialk) + 1;
                    assert(all(S(hkidx(1), :) == initialk(1, :)));
                else
                    S = [initialk; S]; 
                    hkidx = 1:m;
                end
            else
                % Initial activations and nsamp binary samples.
                S = [initialk; randi([0, 1], self.nsamp, self.layers{k}.NumOutputs)];
                hkidx = 1:m;
            end         
            % Select best (from error and proximity to initialk);
            targetk = selecttargetk(self, k, S, hkidx, target);
        end
        % Training multiple layers at once.
        function self = trainlayers(self, layeridxs, data)
           % Update specified layers using gradient descent and backprop on data.
           % Maintains connectivity.
           gradFcn = getGradientFcn(self.TransferFcns(layeridxs), self.ErrorFcns(layeridxs(end)));
           % Perform backprop updates. 
           for i = 1:self.nupdate
               % Compute gradient.
               [~, dE_dW] = feval(gradFcn, self.W(layeridxs), data.input, data.target);
               % Update each layer.
               for j = 1:length(layeridxs)
                   layeridx = layeridxs(j);
                   % Scale so that dE_dW is average per sample.
                   deltaW = - dE_dW{j} / size(data.input, 1);
                   % Scale by layer-specific learning rate.
                   deltaW = self.layers{layeridx}.nu * deltaW;
                   % Modify W. Connectivity enforced in layer object.
                   self.layers{layeridx}.W = self.layers{layeridx}.W  + deltaW;
               end
           end           
        end
        % Saving to file.
        function string = param2str(self)
            % Creates short string describing parameters of netdt.
            % We include ErrorFcns{1}.
            string = sprintf('%s_%s_bs%d_nup%d_nu%f_nsamp%d_nvar%d_%s_%s', ...
                        class(self), arch2str(self), ...
                        self.batchsize, self.nupdate, self.nu, self.nsamp, self.nvarnodes, ...
                        self.ErrorFcns{1}, self.ErrorFcn);
        end
        % Analaysis tools (non-critical)
        function printstatus(self)
            % Simply print error to screen.
            i = length(self.ErrorTrain) - 1; % First is initialization.             
            if ~isempty(self.ErrorTestClass)
                % Train and test class error.
                fprintf('Iteration:%d\tError:%0.15f\tTrainError:%0.06f\tTestError:%0.06f\n', i, self.ErrorTrain(end), self.ErrorTrainClass(end), self.ErrorTestClass(end));
            elseif ~isempty(self.ErrorTrainClass)
                % Train and test class error.
                fprintf('Iteration:%d\tError:%0.15f\tTrainError:%0.06f\n', i, self.ErrorTrain(end), self.ErrorTrainClass(end));
            elseif ~isempty(self.ErrorTest)
                % Train and test error.
                fprintf('Iteration:%d\tError:%0.15f\tTestError:%0.06f\n', i, self.ErrorTrain(end), self.ErrorTest(end));
            else
                % Just train error.
                fprintf('Iteration:%d\tError:%0.15f\n', i, self.ErrorTrain(end));
            end
        end 
        function zeronodes = countZeroNodes(self, data)
            % Returns counts of dead nodes in each layer that
            % have 0 activation for all data.
            X = getActivations(self, data.input);
            zeronodes = zeros(1,self.nlayers);
            for i = 1:self.nlayers
                zeronodes(i) = sum(or(all(X{i} == 0), all(X{i} == 1)));
            end
        end

    end
    % ==================================================
    % PRIVATE METHODS
    % ==================================================
    methods (Access='protected')
        function targetk = selecttargetk(self, k, Sk, hkidx, target)
            % Select target for layer k from set Sk, such that the
            % error to target is minimized.
            % Sk(hdidx) gives the current activation hk. So we can prefer
            % targets close to hk.
            switch self.ErrorFcn
                case {'CrossEntropyError', 'SumSquaredError', 'Hamming', 'L1Error'}
                    targetk = selecttargetk_basic(self, k, Sk, hkidx, target);
                case 'MulticlassCrossEntropyError'
                    targetk = selecttargetk_multiclass(self, k, Sk, hkidx, target);
                otherwise
                    error('Unknown transfer function.')
            end
        end
        function targetk = selecttargetk_basic(self, k, Sk, hkidx, target)
            % Find best target for layer k from set Sk. Breaks ties with hk
            % Compute output for input set.
            output = applyk(self, k, Sk);
            % Find targetk in Sk. Use existing hk to break ties.
            numS = size(Sk,1);
            minidxs = zeros(size(target,1), 1);
            for i = 1:size(target, 1)
                % Error function may not be symmetric.
                errors = feval(self.ErrorFcn_Targets, output, repmat(target(i,:), numS, 1));
                minval = min(errors);
                minset = find(errors == minval);
                if any(minset == hkidx(i))
                    % Use existing target.
                    minidxs(i) = hkidx(i);
                else
                    % Use random target from set of best.
                    minidxs(i) = minset(randi(length(minset)));
                end
            end
            targetk = Sk(minidxs, :);
        end
        function targetk = selecttargetk_multiclass(self, k, Sk, hkidx, target)
            % Find best target for layer k from set Sk. Uses fact that the
            % output is multiclass to speed things up.
            % Find targetk in Sk. Use existing hk to break ties.
            numS = size(Sk, 1);
            [m, nclass] = size(target);
            
            % Compute errors for each sample in Sk and possible class.
            output = applyk(self, k, Sk);
            errors = zeros(numS, nclass);
            for c = 1:nclass
                % Compute error for each input conditional on class being c.
                temptarget = zeros(numS, nclass); temptarget(:, c) = 1;
                errors(:, c) = feval(self.ErrorFcn, output, temptarget);
            end
            
            % Choose best target for each class. (min val in each column of errors)
            minidxc = zeros(nclass, 1);
            for c = 1:nclass
                minerrorc = min(errors(:,c)); % Min error value for class c.
                minsetc = find(errors(:,c) == minerrorc); % Set of idxs for c.
                minidxc(c) = minsetc(randi(length(minsetc))); 
            end
            [classidx, ~] = find(target');            
            targetk = Sk(minidxc(classidx), :);
        end
        function targetk = selecttargetk_single(self, k, Sk, target)
            % Find best target for single example at layer k from set Sk. 
            % Give preference to first row of Sk (initial act). 
            % Compute output for input set.
            assert(size(target, 1) == 1);
            output = applyk(self, k, Sk);
            % Find targetk in Sk. Use existing hk to break ties.
            errors = feval(self.ErrorFcn_Targets, output, repmat(target, size(output,1), 1));
            minval = min(errors);
            minset = find(errors == minval);
            if any(minset == 1)
                % Initial target is as good as any.
                targetk = Sk(1,:);
            else
                % Use random target from set of best.
                targetk = Sk(minset(randi(length(minset)), :), :);
            end
        end
    end
    % ==================================================
    % STATIC METHODS
    % ==================================================
    methods (Static = true, Access='protected')
        function [fullarch, TransferFcns, ErrorFcns] = processArgs(varargin)
            % Process various forms of input for constructor.
            % INPUT: (arch, data)
            %     OR (fullarch, TransferFcns, ErrorFcns)
            % OUTPUT:
            %   fullarch = vector specifying n0,...,nl
            %   TransferFcns = cellarray specifying all transfer functions
            %   ErrorFcns = Functions for training each layer.
            if nargin == 2
                % Convienient constructor from hidden arch and data.
                arch = varargin{1};
                data = varargin{2};
                % fullarch
                if ischar(data), data = loaddata(data); end;
                fullarch = [size(data.input,2), arch, size(data.target,2)];
                % TransferFcns
                [TransferFcns, ErrorFcn] = mlp.detectFcns(fullarch, data);
                % ErrorFcns
                ErrorFcns = repmat({'CrossEntropyError'}, 1, length(fullarch)-1);
                ErrorFcns{end} = ErrorFcn;
            elseif nargin == 3
                % Fully specified.
                fullarch = varargin{1};
                TransferFcns = varargin{2};
                ErrorFcns = varargin{3};
                % If single TransferFcn is given, assume it for all.
                if ~iscell(TransferFcns) || (iscell(TransferFcns) && length(TransferFcns)==1)
                    % Assign this transfer function to every layer.
                    assert(ischar(TransferFcns));
                    TransferFcns = repmat({TransferFcns}, 1, length(fullarch)-1);
                end
                % If single ErrorFcn is given, assume it for all.
                if ~iscell(ErrorFcns) || (iscell(ErrorFcns) && length(ErrorFcns)==1)
                    % Assign this ErrorFcn function to every layer.
                    assert(ischar(ErrorFcns));
                    ErrorFcns = repmat({ErrorFcns}, 1, length(fullarch)-1);
                end
            else
                error('Input error.')
            end
        end
    end
end


