function E = meanMulticlassCrossEntropyError(X, target)
% Cross entropy error for multiclass output. Where sum of activation is 1.

% Might have to handle log(0).
E = - mean( sum(target .* log(X), 2) ); 

if any(isnan(E))
    keyboard
    % Machine precision hack.
    epsilon = 1e-45; % To avoid log(1e-46)=-inf
    errors = - mean(sum( target .* log(max(output, epsilon)), 2));
end
end