classdef netdt_general < netdt_limited 
    %  Generalization of netdt for updating multiple layers at once.
    properties
        schedule = {};  % Cell array specifying how to update layers.
    end
    methods
        function self = netdt_general(connectivity, varargin)
            % Constructor is same as for netdt_limited.
            self = self@netdt_limited(connectivity, varargin{:});
            self.schedule = createSchedule(self.nlayers);
        end
        function self = train(self, data)
            % Train network.
            % Get batchdata.
            if ischar(data), self.dataset=data; data = loaddata(data); end;
            batchdata = makeBatchData(data, self.batchsize);
            nbatch = length(batchdata);
            % Set learning rates at each layer. Scale nu by number of inputs.
            for k = 1:self.nlayers
                self.layers{k}.nu = self.layers{k}.calcAdjustednu(self.nu);
            end
            % Record initial error.
            if isempty(self.ErrorTrain)
                self = recordErrors(self, data);
                printstatus(self);
            end
            % Compute initial activations.
            X = getActivations(self, batchdata{1}.input);
            % Begin learning
            for i = 1:self.nepoch
                for j = 1:nbatch
                    for kk = 1:length(self.schedule)
                        % Determine layers to be trained.
                        klayers = self.schedule{kk};
                        k1 = klayers(1);
                        k2 = klayers(end);
                        
                        % Find layerdata.input.
                        if k1==1
                            layerdata.input = batchdata{j}.input;
                        else
                            layerdata.input = X{k1-1}; % Should be updates.
                        end
                        
                        % Find layerdata.target.
                        initialk2 = applyk1k2(self, k1, k2, layerdata.input); 
                        layerdata.target = optimizeDeepTargets(self, k2, initialk2, batchdata{j}.target);

                        % Update.
                        self = trainlayers(self, klayers, layerdata);
                        
                        % Compute output of these layers. Used for next layer
                        X(k1:k2) = getActivationsk1k2(self, layerdata.input, k1, k2);                
                    end %layer
                end % batch
                % Record errors
                self = recordErrors(self, data);
                printstatus(self)
            end % epoch
        end
    end
end

function schedule = createSchedule(nlayers)
% Create a default schedule cell array.
assert(mod(nlayers,2) == 0);
schedule = cell(1, (nlayers)/2);
for i = 1:(nlayers)/2
    idx = (i-1)*2+1;
    schedule{i} = idx:idx+1; 
end
end