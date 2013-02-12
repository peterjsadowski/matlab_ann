clear all
maxNumCompThreads(16);

% data = loaddata('mnistauto');
data = loaddata('mnistclassify');

n = netbackgen([100], data);
% n = netbackgen2([500 100 500], data);
% n = netbackgen2([1000 500 100 500 1000], data);


% %n = convert2threshold(n);
% % Convert hidden layers to threshold.
% for i = 1:n.nlayers -1
%     n.layers{i}.TransferFcn = 'threshold';
%     n.layers{i}.ErrorFcn = 'Hamming';
% end
% % n.ErrorFcn = 'Hamming';
% % % Last layer
% n.layers{end}.ErrorFcn = 'CrossEntropyError'; % 
% n.ErrorFcn = 'CrossEntropyError';

n.initscale = 1;
n = initialize(n);
%n.mu = 1e-6; %1e-6; %0.01;
% n.momentum = 0;
n.nu = 1e-3;%1e-6; %1e-5;

n.batchsize = 10;
n.nupdate = 3;
n.nepoch = 100;

n
n = train(n, data);

% % Save results to file.
% resultsdir = 'results';
% [~,~,~] = mkdir(resultsdir);
% save([resultsdir, '/', n.param2str], 'n');

% Analysis
% X = getActivations(n, data.input);
% imagesc([X{1}, X{2}, X{end}])