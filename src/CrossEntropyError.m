function E = CrossEntropyError(X, target)
% Cross entropy error for independent outputs.

% Might have to handle log(0).
E = - sum( target .* log(X) + (1-target) .* log(1-X) , 2); 

if any(isinf(E)) || any(isnan(E))
    %keyboard
    % Machine precision hack.
    %warning('Avoiding machine precision problems with log(0)')
    epsilon = 1e-45; % To avoid log(1e-46)=-inf
    E = -sum( target .* log(max(X, epsilon)) + (1-target) .* log(max(1-X, epsilon)), 2);
end
end