function GradFcn = getGradientFcn(TransferFcns, ErrorFcn)
% Determine gradient fcn from specified transfer functions and error function.
% TransferFcns = Cell array, specifying transfer function for each layer.
% ErrorFcn = Error function for output layer.

if strcmpi(TransferFcns{end}, 'logsig') && strcmpi(ErrorFcn, 'CrossEntropyError')
        % Logistic in every layer.
        GradFcn = 'gradient_crossent';
        for k = 1:length(TransferFcns)-1
            assert(strcmp('logsig', TransferFcns{k}));
        end
elseif strcmpi(TransferFcns{end}, 'SoftmaxTransfer') && strcmpi(ErrorFcn, 'MulticlassCrossEntropyError')
        % Logistic in every layer but last, for multi class classification.
        GradFcn = 'gradient_softmaxcrossent';
        for k = 1:length(TransferFcns)-1
            assert(strcmp('logsig', TransferFcns{k}));
        end
elseif strcmpi(TransferFcns{end}, 'linear') && strcmpi(ErrorFcn, 'SumSquaredError')
        % Logistic transfer in every layer but last, which is linear.
        GradFcn = 'gradient_regression';
        for k = 1:length(TransferFcns)-1
            assert(strcmp('logsig', TransferFcns{k}));
        end
else
    error('Unknown transfer-error pair.');
end
end