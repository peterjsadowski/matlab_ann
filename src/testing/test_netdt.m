clear all
%addpath(genpath('/extra/pjsadows0/ml/ann-prototypes/src/')); 

data = loaddata('mnistclassify');
n = netdt([100 100], data);

for i = 1:n.nlayers - 1
    n.layers{i}.TransferFcn = 'threshold';
    n.layers{i}.ErrorFcn = 'Hamming';
end

% Convert to threshold.
% n = convert2threshold(n);
% n.ErrorFcn = 'L1Error';
% n.ErrorFcn = 'SumSquaredError';
% n.ErrorFcns = n.ErrorFcn;

n.initscale = 1;
n = initialize(n);
n.nepoch = 1; %200;
n.nsamp = 100;
n.nupdate = 10;
n.nu = 0.01;
n.batchsize = 1000;
n.nvarnodes = 3;

n
n = train(n, data);

%X = getActivations(n, data.input);
%imagesc([data.input, X{1}, X{2}, X{end}, data.target])
% imagesc(X{1}>=0.5);


