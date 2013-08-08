classdef netsemiauto < mtmlp
% MLP trained as a stack of mtmlp.
% Assumes input is [0,1].
    properties
        ninitialepochs = 0 % Number of initialization epochs.
    end
    
    properties (Dependent = true)
        alphas
    end
    properties (Access = private)
        alpha_list = []; % Should only use 'alphas' property. 
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        function self = netsemiauto(varargin)
            % Initialize first part of the network just as we do for mtmlp.
            self = self@mtmlp(varargin{:});
            self.alphas = zeros(1, self.nlayers);
        end
        function self = set.alphas(self, alphas)
            % Set alpha parameter value of each layer: 1,2,3...nlayers.
            assert(length(alphas) == self.nlayers);
            self.alpha_list = alphas;
            self.alpha = alphas(end);
        end
        function alphas = get.alphas(self)
            % Return alpha parameter value for each layer 1,2,3,etc.
            self.alpha_list(end) = self.alpha; % In case someone changed alpha property.
            alphas = self.alpha_list;
        end
        function self = train(self, data)
            % Train each layer, starting from bottom. This is recursive.
            if self.nlayers > 1
                % Recursion step:
                % Create a new netsemiauto, with one less layer.
                tempnet = getLowerNet(self);
                % Train it.
                tempnet = train(tempnet, data);
                % Update weights: take from tempnet.
                self.W(1:end-2) = tempnet.W(1:end-1);
            end
            % Train multitask network.
            fprintf('\nTraining layer %d. alpha=%d\n', self.nlayers, self.alpha);
            assert(self.alphas(end) == self.alpha);
            
            % Train only the output weights to get things started.
            assert(~self.limitbackprop1)
            self.limitbackprop1 = true;
            fprintf('Fixing all but output weights for the first epoch...\n')
            temp = self.nepoch - self.ninitialepochs;
            self.nepoch = self.ninitialepochs;
            %self = updateWeights(self, data);
            self = train@mtmlp(self, data);
            self.nepoch = temp;
            fprintf('...done fixing.\n')
            self.limitbackprop1 = false;
            
            % Train the rest.
            self = train@mtmlp(self, data);
        end
        function n = getLowerNet(self)
            % This function returns a new network. The weights and
            % properties are identical to the source network, except that
            % the top hidden layer has been removed.
            % Keep: fullarch, TransferFcns, ErrorFcn, W, parameters.
            fullarch = self.fullarch([1:end-2, end]); % God I love matlab.
            TransferFcns = self.TransferFcns([1:end-2, end]); 
            ErrorFcn = self.ErrorFcn;
            n = netsemiauto(fullarch, TransferFcns, ErrorFcn);
            % Default initialization. Use same training parameters.      
            n.nepoch    = self.nepoch;   
            n.batchsize = self.batchsize;     
            n.nupdate   = self.nupdate;
            n.debug     = self.debug;
            n.limitbackprop = self.limitbackprop;
            n.limitbackprop1= self.limitbackprop1;
            n.convflag      = self.convflag;
            n.convthreshold = self.convthreshold;
            n.convwindow    = self.convwindow;
            n.convmin       = self.convmin;
            % Alphas
            n.alphas        = self.alphas(1:end-1);
            % For lower layers of n, use self weights. (Might already be trained.)
            n.W(1:end-1) = self.W(1:end-2);
        end
            
        % Output
        function string = param2str(self)
            % Creates short string describing parameters of netdt.
            if self.convflag
                time = clock;
                string = sprintf('semiauto_%s_bs%d_nup%d_nep%d_limit%d_alphas%s_ct%d_day%dhr%dmin%dsec%d', ...
                            arch2str(self), ...
                            self.batchsize, self.nupdate, self.nepoch, ...
                            self.limitbackprop, alphas2str(self),...
                            -log10(self.convthreshold),...
                            time(3), time(4), time(5), floor(time(6)));
            else
                time = clock;
                string = sprintf('semiauto_%s_bs%d_nup%d_nep%d_limit%d_alphas%s_day%dhr%dmin%dsec%d', ...
                            arch2str(self), ...
                            self.batchsize, self.nupdate, self.nepoch, ...
                            self.limitbackprop, alphas2str(self),...
                            time(3), time(4), time(5), floor(time(6)));
            end
        end
        function string = alphas2str(self)
            string = sprintf('_%0.0f', 10*self.alphas);
        end
        
    end
    
    


end

