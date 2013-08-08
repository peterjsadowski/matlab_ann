addpath(genpath('/extra/pjsadows0/ml/ann-prototypes/src'))
clear all
%maxNumCompThreads(16);

dataname = 'mnist';
% dataname = 'test';

data = loaddata(dataname);
% data.input = data.input(1:10,:);
% data.target = data.target(1:10,:);
% data = loaddata('mnistclassify');
% data.target = data.target >= 0.5;
% data.testtarget = data.testtarget >= 0.5;
 
n = netbp_dropout([800 800], data);
%n.dropprob = [0, 0 , 0];
n.dropprob = [.2, .5, .5];

% n = netbp_dropout([100], data);
% n.dropprob = [0 0];

n.initscale = 1;
n = n.initialize();
% n = n.initializeBias(-.25);

n.batchsize = 100; 
n.nu = 10; %1e-6; %1e-5; % Grad is scaled by batchsize.
n.mu = .5;
n.nepoch = 100;
dataname
n
% n = train(n, data);
%savenet(n, ['results/', dataname, '/'])

n.nu=10; n.mu=.5; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=9; n.mu=.6; n = train(n, data); savenet(n, ['~/temp/', dataname, '']);
n.nu=8; n.mu=.7; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=7; n.mu=.8; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=6; n.mu=.9; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=5; n.mu=.99; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=4; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=3; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=2; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=1; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=.5; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);
n.nu=.1; n = train(n, data); savenet(n, ['~/temp/', dataname, '/']);



% % Save results to file.
% resultsdir = 'results/exp1';
% [~,~,~] = mkdir(resultsdir);
% save([resultsdir, '/', n.param2str], 'n');
% 
% % Analysis
% X = getActivations(n, data.input);
% imagesc([X{1}, X{2}, X{end}])
