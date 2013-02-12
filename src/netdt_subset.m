classdef netdt_subset < netdt
% Net trained with deep target algorithm, update subset in each layer.
    properties
        nsubset = 10; % Update this number of nodes in each layer.
    end
    methods
        function self = netdt_subset(varargin)
            self = self@netdt(varargin{:});
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
            % Initial activations.
            X = cell(1, self.nlayers);
            % Begin learning
            for i = 1:self.nepoch
                for j = 1:nbatch
                    for k = 1:self.nlayers
                        % Find layerdata.input.
                        if k==1
                            layerdata.input = batchdata{j}.input;
                        else
                            layerdata.input = X{k-1}; % Should be updates.
                        end
                        
                        % Last layer is standard update. No dropout.
                        if k == self.nlayers
                            layerdata.target = batchdata{j}.target;
                            self.layers{k} = train_gd(self.layers{k}, layerdata, self.nupdate);
                            X{k} = apply(self.layers{k}, layerdata.input);
                            continue
                        end
                        
                        % Choose dropout nodes (subset=notdrop).
                        notdropidx = randperm(self.layers{k}.NumOutputs, self.nsubset);
                        dropidx = setdiff(1:self.layers{k}.NumOutputs, notdropidx);
                        
                        % Find layerdata.target.
                        initialk = apply(self.layers{k}, layerdata.input);
                        layerdata.target = optimizeTargets(self, k, initialk, batchdata{j}.target, dropidx);
                        
                        % Update. Does not update input weights to dropped
                        self.layers{k} = train_gd_dropout(self.layers{k}, layerdata, self.nupdate, notdropidx);
                        
                        % Compute output of this layer. Used for next layer
                        X{k} = apply(self.layers{k}, layerdata.input);
                    end %layer
                end % batch
                % Record errors
                self = recordErrors(self, data);
                printstatus(self)
            end % epoch
        end
        function targetk = optimizeTargets(self, k, initialk, target, dropidx)
            % Optimize targets for layer k of architecture, with dropped
            % nodes.
            % k = layer idx -- we need target for this layer.
            % initialk = m x n_(k) matrix of layer k activation hk.
            % target = target for layer l.
            assert(k ~= self.nlayers, 'Should not be dropping output nodes.');
            % Number of nodes that are not dropped.
            notdropidx = setdiff(1:self.layers{k}.NumOutputs, dropidx);
            numnotdrop = length(notdropidx);
            % Initial activation of non dropped nodes.
            initialk(:, dropidx) = 0;
            m = size(target, 1);
            % Find sample set S.
            if numnotdrop <= 10
                % Sample all binary targets over non dropped nodes.
                keyboard % No easy way to do this.
                S = zeros(2^numnotdrop, self.layers{k}.NumOutputs);
                S(:, notdropidx) = de2bi(0:(2^numnotdrop - 1));
                % If real valued output, include initial activation too
                if strcmp(self.layers{k}.TransferFcn, 'threshold')
                    % Use trick to find hkidx s.t. S(hkidx) = initialk. 
                    hkidx = bi2de(initialk(:, notdropidx)) + 1;
                    assert(all(S(hkidx(1), :) == initialk(1, :)));
                else
                    S = [initialk; S];
                    hkidx = 1:m;
                end
            else
                % Initial activations and nsamp binary samples.
                S = [initialk; randi([0, 1], self.nsamp, self.layers{k}.NumOutputs)];
                hkidx = 1:m;
                S(:, dropidx) = 0;
            end
            % Scale up by dropout factor.
            factor = (self.layers{k}.NumOutputs / numnotdrop);
            S = S * factor;
            % Select best (from error and proximity to initialk);
            targetk = selecttargetk(self, k, S, hkidx, target);
            % Remove factor.
            targetk = targetk / factor;
        end
    end
end