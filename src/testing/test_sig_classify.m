clear all
%maxNumCompThreads(16);

dataname = 'mnistclassify1000';
% dataname = 'mnistautobinary1000';

% dataname = 'mnistclassify';
data = loaddata(dataname);
% data.input = data.input(1:10,:);
% data.target = data.target(1:10,:);
% data = loaddata('mnistclassify');
% data.target = data.target >= 0.5;
% data.testtarget = data.testtarget >= 0.5;

% n = netdt([1000 500 100 500 1000], data);
% n = netdt([1000 500], data);
% n = netdt([100 100], data);
% n = netdt([2000], data);
% n = netdt([1000 1000 1000 1000 1000], data);
n = netdt(300*ones(1,20), data);
% n = netdt([500, 300], data);
% n = netdt([1000 1000], data);
% 
% % Convert hidden layers to threshold.
% for i = 1:n.nlayers -1
%     n.layers{i}.TransferFcn = 'threshold';
%     n.layers{i}.ErrorFcn = 'Hamming';
% end
% % % % % Last layer
% n.layers{end}.ErrorFcn = 'MulticlassCrossEntropyError'; % 
% n.layers{end}.TransferFcn = 'SoftmaxTransfer';
% n.ErrorFcn = 'MulticlassCrossEntropyError';
% n.layers{end}.TransferFcn = 'threshold';
% n.layers{end}.ErrorFcn = 'Hamming';teset
% n.ErrorFcn = 'Hamming';

% n = convert2threshold(n);
% n.ErrorFcn_Targets = 'Hamming';  % Speeds up target finding
% n.ErrorFcn_Targets = 'SumSquaredError';  % Speeds up target finding

n.initscale = 1;
n = n.initialize();
% n = n.initializeBias(-.25);
% prev = load('results/netdt_dropout_784_1000_500_100_10_bs100_nup10_nsamp0_Hamming_MulticlassCrossEntropyError_error0.mat');
% n.W = prev.n.W;

% n.opttargetfcn = 'optimizeTargets_sparse';
n.opttargetfcn = 'optimizeTargets_local';
n.nvarnodes = 5; % 10;
% n.schedule = 3;

% n.opttargetfcn = 'optimizeTargets';
% n.nvarnodes = Inf;

n.nsamp = 100;
n.batchsize = 100;

% n.layertrainfcn = 'train_gd_ipocket'; % Independent pockets
% n.nupdate = 1; % maxnum

n.layertrainfcn = 'train_gd_ipocket'; % Independent pockets
n.nupdate = 1; % maxnum
n.nu = 1e-2; %1e-6; %1e-5; % Doesn't matter for all threshold.
%n.mu = 0.0001;
%n.momentum = 0;

n.nepoch = 1000;

dataname
n
n = train(n, data);
savenet(n, ['results/', dataname, '/'])

% % Save results to file.
% resultsdir = 'results/exp1';
% [~,~,~] = mkdir(resultsdir);
% save([resultsdir, '/', n.param2str], 'n');
% 
% % Analysis
% X = getActivations(n, data.input);
% imagesc([X{1}, X{2}, X{end}])
