function E = MulticlassCrossEntropyError(X, target)
% Cross entropy error for multiclass output. Where sum of activation is 1.
assert(all(max(target, [], 2) == 1), 'Make sure target is correct. Order matters.')

% Might have to handle log(0).
E = - sum(target .* log(X), 2); 

if any(isnan(E))
    % Machine precision hack.
    epsilon = 1e-45; % To avoid log(1e-46)=-inf
    E = - sum( target .* log(max(X, epsilon)), 2);
end
end