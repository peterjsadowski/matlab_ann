

data = loaddata('testnoisy');
p = perceptron(100, 10, 'softmax', 'MulticlassCrossEntropyError'); p = train_gd(p, data, 100);
clear m; m = netdt([100 10], {'softmax'}, {'MulticlassCrossEntropyError'}); m.nu=1; m.nepoch=100; m = train(m, data);
clear m; m = netbp([100 10], {'softmax'}, {'MulticlassCrossEntropyError'}); m.nepoch=100; m = train(m, data);

clear m; m = netbp([100 10 10], {'logsig', 'softmax'}, 'MulticlassCrossEntropyError'); m = train(m, data);
clear m; m = netdt([100 10 10], {'logsig', 'softmax'}, {'CrossEntropyError', 'MulticlassCrossEntropyError'}); m = train(m, data);

data = loaddata('testnoisyauto');
clear m; m = netbp([100 10 100], 'logsig', 'CrossEntropyError'); m = train(m, data);
clear m; m = netdt([100 30 10 30 100], 'logsig', 'CrossEntropyError'); m.nepoch=2; m = train(m, data);
% Now change this to a threshold gate network.
m.TransferFcns = 'threshold'; % Change all transfer fcns.
m.ErrorFcns = 'Hamming'; 
m.ErrorFcn = 'Hamming';
test(m, data)
m = train(m, data);


% data = loaddata('testauto');
% clear m; m = netbp([100 10 100], 'logsig', 'CrossEntropyError'); m = train(m, data);
