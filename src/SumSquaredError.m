function E = SumSquaredError(X, target)
% Sum of squares error function.
E = sum( (X - target).^2 , 2);
end