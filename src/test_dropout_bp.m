clear all
%maxNumCompThreads(16);

% dataname = 'mnistclassify';
dataname = 'mnistclassify';
% dataname = 'test';

% dataname = 'mnistclassify';
data = loaddata(dataname);
% data.input = data.input(1:10,:);
% data.target = data.target(1:10,:);
% data = loaddata('mnistclassify');
% data.target = data.target >= 0.5;
% data.testtarget = data.testtarget >= 0.5;

% n = netbp_dropout([100], data);
% n.dropprob = [.2, .5];
% 
n = netbp_dropout([100 100], data);
n.dropprob = [0, 0 , 0];
% n.dropprob = [.2, .5, .5];
% n = netbp_dropout([100 100 100], data);
% n.dropprob = [.2, .5, .5 .5];
% n = netbp_dropout([500 300 100 100], data);
% n.dropprob = [.2, .5, .5 .5 .5];

% n = netbp_dropout([100], data);
% n.dropprob = [0 0];


n.initscale = 1;
n = n.initialize();
% n = n.initializeBias(-.25);

n.batchsize = 1;
n.nu = 1e-2; %1e-6; %1e-5; % Doesn't matter for all threshold.
n.nepoch = 500;

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
