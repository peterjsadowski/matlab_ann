clear all
maxNumCompThreads(16);


% data = loaddata('testnoisyauto');
% n = netdt([50 30 20 30 50], data);

data = loaddata('mnistauto');
% data.target = data.target >= 0.5;
% data.testtarget = data.testtarget >= 0.5;

% n = netdt([1000 500 100 500 1000], data);
n = netdt([1000 100 1000], data);
%  n = netdt([100], data);
% n = convert2threshold(n);
% Convert hidden layers to threshold.
for i = 1:n.nlayers-1
    n.layers{i}.TransferFcn = 'threshold';
    n.layers{i}.ErrorFcn = 'Hamming';
end
% Last layer
n.layers{end}.ErrorFcn = 'CrossEntropyError'; % 
n.ErrorFcn = 'CrossEntropyError';
n.ErrorFcn_Targets = 'SumSquaredError';  % Speeds up target finding

% n.ErrorFcn = 'L1Error';
% n.ErrorFcn = 'SumSquaredError';
% n.ErrorFcns = n.ErrorFcn;

n.initscale = 1;
n = initialize(n);
n.nepoch = 10;
n.nupdate = 5;
n.nu = 1e-5;%1e-6; %1e-5;
%n.mu = 0.01;
n.momentum = 0.9;
n.batchsize = 100;
n.nsamp = 100;

n
n = train(n, data);

% Save results to file.
resultsdir = 'results/exp1';
[~,~,~] = mkdir(resultsdir);
save([resultsdir, '/', n.param2str], 'n');

% Analysis
X = getActivations(n, data.input);
imagesc([X{1}, X{2}, X{end}])
