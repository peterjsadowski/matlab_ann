function E = MulticlassZeroOneError(X, target)
% Compute zero-one error for multiclass output. Ties broken with ordering.
% INPUT: X = mxc matrix of real values summing to 1.
%        target = mxc matrix of target classes (all zeros but 1 in each row)
assert(length(unique(target(:))) == 2); % Binary.
assert(all(sum(target, 2) == 1));
[targetidx, ~] = find(target');
assert(all(targetidx <= size(target, 2)));
[~, Xidx] = max(X, [], 2);
E = (targetidx ~= Xidx);
end