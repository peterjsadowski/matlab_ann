clear all

% data = loaddata('testnoisyauto');
% n = netdt([50 30 20 30 50], data);

data = loaddata('mnistauto');
% data.target = data.target >= 0.5;
% data.testtarget = data.testtarget >= 0.5;

% n = netdt([1000 100 1000], data);
n = netbp([1000 100 1000], data);
% Convert to threshold.
% n = convert2threshold(n);
% for i = 1:n.nlayers
%     n.layers{i}.TransferFcn = 'threshold';
%     n.layers{i}.ErrorFcn = 'Hamming';
% end

% n.ErrorFcn = 'L1Error';
% n.ErrorFcn = 'SumSquaredError';
% n.ErrorFcns = n.ErrorFcn;

n.initscale = 1;
n = initialize(n);
n.nepoch = 20;
% n.nupdate = 50;
% n.nu = 0.02;
n.batchsize = 1000;

n
n = train(n, data);

% Convert hidden layers to threshold
nt = n;
for i = 1:nt.nlayers-1
    nt.layers{i}.TransferFcn = 'threshold';
    nt.layers{i}.ErrorFcn = 'Hamming';
end
nt = recordErrors(nt, data);
printstatus(nt);


% X = getActivations(n, data.input);
% imagesc([data.input, X{1}, X{2}, X{end}, data.target])

% Save results to file.
resultsdir = 'results/exp1';
[~,~,~] = mkdir(resultsdir);
save([resultsdir, '/', n.param2str], 'n');

