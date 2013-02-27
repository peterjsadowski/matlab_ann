function error = meanHamming(X, target)
assert(all(size(X) == size(target)));
error = biterror(X, target) / size(X,1);
end