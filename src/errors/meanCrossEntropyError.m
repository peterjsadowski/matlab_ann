function E = meanCrossEntropyError(X, target)
% Cross entropy error, where 

% Might have to handle log(0).
E = - mean( sum( target .* log(X) + (1-target) .* log(1-X) , 2) ); 

if any(isnan(E))
    keyboard
    % Machine precision hack.
    epsilon = 1e-45; % To avoid log(1e-46)=-inf
    errors = -sum( target .* log(max(output, epsilon)) + (1-target) .* log(max(1-output, epsilon)), 2);
end
end