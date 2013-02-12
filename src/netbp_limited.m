classdef netbp_limited < netbp
    % Network with limited connectivity.
    methods
        function self = netbp_limited(connectivity, varargin)
            % Create mlp with layers of limited connectivity. 
            self = self@netbp(varargin{:});
            % Initialize each perceptron with limited connectivity.
            if (length(connectivity) == 1);
                % Every unit in every layer has specified number of inputs,
                % plus bias term.
                connectivity = repmat({connectivity}, 1, self.nlayers);
            end
            for i = 1:self.nlayers
                self.layers{i} = initialize(self.layers{i}, connectivity{i});
            end 
           
        end
    end
end