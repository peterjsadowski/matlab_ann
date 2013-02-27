function E = meanSumSquaredError(X, target)
% Sum of squares error function.
E = mean(sum( (X - target).^2 , 2));
end