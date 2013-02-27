clear all

% dataname = 'mnistclassify1000';
%dataname = 'mnist_semi_1000';
dataname = 'iris2class';
% dataname = 'mnistclassify1000';
% dataname = 'test';
data = loaddata(dataname);

% Initialize 
% arch = [30 20];
arch = [300 100];
% arch = [500 100];
n = mtmlp(arch, data);
n = n.initialize();
% n.alpha = .1;
n.alpha = 0;
n.nepoch = 100;
n.batchsize = 1000;
n.nupdate = 3; 
n.limitbackprop = false;
n.convflag = true;
n.convthreshold = 1e-5; %
n.convwindow    = 1;  % Convergence window.
n.convmin       = 10;   % Minimum number of iterations before convergence possible.
n.debug = false;
n = train(n,data);
