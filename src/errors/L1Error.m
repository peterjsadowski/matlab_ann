function E = L1Error(X, target)
% Sum of squares error function.
E = sum( abs(X - target) , 2);
end