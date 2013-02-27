classdef netdt_limited < netdt
    % Network with limited connectivity, trained with deep target algorithm.
    methods
        function self = netdt_limited(connectivity, varargin)
            % Create mlp with layers of limited connectivity.
            self = self@netdt(varargin{:});
            if isempty(connectivity)
                % Fully connected.
                for i = 1:self.nlayers
                    self.layers{i} = initialize(self.layers{i});
                end
                return
            elseif (length(connectivity) == 1);
                % Every unit in every layer has specified number of inputs,
                % plus bias term.
                connectivity = repmat({connectivity}, 1, self.nlayers);
            end
            % Initialize each perceptron with limited connectivity.
            for i = 1:self.nlayers
                self.layers{i} = initialize(self.layers{i}, connectivity{i});
            end
        end
    end
end
