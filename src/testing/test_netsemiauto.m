% Path to NN code.
path('/home/pjsadows/ml/prototyping/src/', path); 
clear all

dataname = 'test';
% dataname = 'mnistclassify1000';
data = loaddata(dataname);

%% Single layer network.
n = netsemiauto([], data)
n.nepoch = 20;
n.convflag = true;
n = train(n, data);
savenet(n, ['results/', dataname, '/']);

%% Multilayer network.
n = netsemiauto([80 60 40], data)
n = train(n, data);

%% Deep network. 
n = netsemiauto(20 * ones(1,10), data)
n = train(n, data);

%% Adjust parameters.
n = netsemiauto([80 60 40], data);
n.batchsize = 1000;
n.nupdate = 2;
n.nepoch = 5;
n
n = train(n, data);

%% Adjust alphas.
n = netsemiauto([80 60 40], data);
n.alphas = [1, .6, .3, .1];
n
n = train(n, data);

%% Limit backprop
n = netsemiauto([80 60 40], data);
n.alphas = [.9, .6, .3, .1];
n.limitbackprop = true;
n.debug = true
n.nepoch = 3;
n
n = train(n, data);

%% MNIST
clear all
dataname = 'mnistclassify10000';
data = loaddata(dataname);

n = netsemiauto([200 100 50], data);
n = initialize(n);
n.alphas = [.9, .6, .4, 0];
n
n = train(n, data);

%% MNIST semisupervised
clear all
dataname = 'mnist_semi_10000';
data = loaddata(dataname);

n = netsemiauto([50 10], data);
n = initialize(n);
n.nepoch = 5;
% n.debug = true;
% n.alphas = [.9, .6, .4, 0];
n.alphas = [.9, .45, 0];
n
n = train(n, data);



