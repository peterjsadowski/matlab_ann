clear all
addpath(genpath('../'));


load('../data/mnist/mnist1000.mat'); % Loads struct 'data'

% Initialize network object.
arch = [200 100]; % Hidden layer sizes.
n = netsemiauto(arch, data);

% Constant parameters.
n.batchsize = 10000;    % Number of examples in each batch.
n.nupdate = 3;          % Number of linesearches for each batch.
n.debug = false;
n.convflag      = true; % If true, stop when convergence criteria are met.
n.convthreshold = 1e-10;%
n.convwindow    = 1;    % Convergence window.
n.convmin       = 10;   % Minimum number of iterations before convergence possible.

% Varying parameters.
n.nepoch = 10;                    % Number of epochs.
n.alphas = .9:-.9/length(arch):0; % Smooth transition.
n.limitbackprop = false;          % Do not limit backprop to 1 layer.

% Train
n = train(n, data);
savenet(n, ['results/', dataname, '/']); % Saves trained net object to file.