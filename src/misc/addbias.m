function X = addbias(X)
X = [X, ones(size(X,1),1)];
end