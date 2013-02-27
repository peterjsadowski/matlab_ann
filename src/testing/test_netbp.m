clear all
%maxNumCompThreads(16);

% dataname = 'test';
dataname = 'mnistclassify1000';
% dataname = 'mnistautobinary1000';

% dataname = 'mnistclassify';
data = loaddata(dataname);

n = netbp([300 200 100], data);
% n = netdt([1000 1000], data);
% 

n.initscale = 1;
n = n.initialize();
% n = n.initializeBias(-.25);
% prev = load('results/netdt_dropout_784_1000_500_100_10_bs100_nup10_nsamp0_Hamming_MulticlassCrossEntropyError_error0.mat');
% n.W = prev.n.W;

n.batchsize = 1000;
n.nupdate = 3; % maxnum
n.nepoch = 100;

dataname
n
n = train(n, data);
%savenet(n, ['results/', dataname, '/'])

% % Save results to file.
% resultsdir = 'results/exp1';
% [~,~,~] = mkdir(resultsdir);
% save([resultsdir, '/', n.param2str], 'n');
% 
% % Analysis
% X = getActivations(n, data.input);
% imagesc([X{1}, X{2}, X{end}])
