
% data = loaddata('testnoisy');
% fullarch = [];

% Backprop


% DT
n = netdt([30 10 30], 'testnoisyauto');
n.nepoch = 1000;
n = train(n, 'testnoisyauto');
