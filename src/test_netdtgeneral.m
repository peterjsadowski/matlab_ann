
% n = netdt_limited(2, [50 25], 'test');
% n.nepoch = 1000;
% n = train(n, 'test');
% imagesc(n.W{1});

clear all
% data = loaddata('parity8');
% % n = netdt_general([], [32, 16, 16, 8, 8], data);
% connectivity =  getParityConnectivity(8);
% % n = netdt_general(connectivity, [16, 4, 8, 2, 4], data);
% % n = netdt_general(connectivity, [32, 4, 16, 2, 8], data);
% n = netdt_general(connectivity, [32, 16, 16, 8, 8], data);


data = loaddata('parity16');
n = netdt_general([], [64, 32, 32, 16, 16, 8, 8], data);

% data = loaddata('parity4');
% connectivity =  getParityConnectivity(4);
% n = netdt_general(connectivity, [4, 2, 2], data);
%n = netdt_general([], [16, 8, 4], data); % works on parity4!
% n = netbp_limited(connectivity, [4, 2, 2], data);
% n = netbp([16 8 4], data);

% data = loaddata('parity');
% connectivity = 2;
% n = netdt_general(connectivity, [64, 32, 16, 8, 4, 2], data);
% % % n = netbp_limited(connectivity, [64, 32, 16, 8, 4, 2], data);

%n = netbp_limited(2, [4 2], data);

% % data = loaddata('xor');
% data.input = [0,0;0,1;1,0;1,1];
% data.target = [0;1;1;0];
% % n = netdt_limited([], [2], data);
% n = netdt_general([], [2], data);
% % n = netbp(2, data);
% % data.target = and(data.input(:,1), data.input(:,2));


% data = loaddata('testnoisy');
% n = netdt([30 10 30], data);

% Convert to threshold.
% n = convert2threshold(n);

% n.ErrorFcn = 'L1Error';
% n.ErrorFcn = 'SumSquaredError';
% n.ErrorFcns = n.ErrorFcn;

% n = initialize(n);
n.nepoch = 1000;
n.nupdate = 10;
n.nu = 0.5;
n.batchsize = 10;

n = train(n, data);

X = getActivations(n, data.input);
imagesc([data.input, X{1}, X{2}, data.target])
% imagesc(X{1}>=0.5);


