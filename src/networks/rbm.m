% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   

% Modified by Peter Sadowski October 2010

% The program takes input:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning
% If restart == 0, then we also need to specify the starting values for:
% vishid    -- Weight matrix for these two layers.
% hidbiases -- Biases for hidden layer.
% visbiases -- Biases for visible layer.

classdef rbm < perceptron
    properties
        % Default learning parameters used by Hinton and Salakhutdinov.
        epsilonw      = 0.1;   % Learning rate for weights
        epsilonvb     = 0.1;   % Learning rate for biases of visible units
        epsilonhb     = 0.1;   % Learning rate for biases of hidden units
        weightcost    = .0002;  % weight cost  = 0.0002 in Hinton's code.
        initialmomentum  = 0.5;
        finalmomentum    = 0.9;
        batchsize = 1000;
        
        % Visual biases. (Normal perceptron doesn't have this.
        visbias
       
    end
    % ==================================================
    % PUBLIC METHODS
    % ==================================================
    methods
        function self = rbm(nvis, nhid)
            % nvis = Units in visible (input) layer.
            % nhid = Units in hidden (output) layer.
            self = self@perceptron(nvis, nhid, 'logsig', 'CrossEntropyError');
        end
        function self = initialize(self)
            self = initialize@perceptron(self);
            self.visbias = zeros(1, self.NumInputs);
        end
        function self = train(self, data, maxepoch)
            % Train weights using single-step Contrastive Divergence.
            batchdata = makeBatchData(data, self.batchsize);

            numhid = self.NumOutputs;
            numdims = self.NumInputs;
            vishid = self.W(1:end-1,:);
            hidbiases = self.W(end,:);
            visbiases = self.visbias;
            
            % Initialize increment matrices.
            vishidinc  = zeros(numdims,numhid);
            hidbiasinc = zeros(1,numhid);
            visbiasinc = zeros(1,numdims);
            for epoch = 1:maxepoch,
                errsum=0;
                for batch = 1:length(batchdata)
                    %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    data = batchdata{batch}.input;
                    numcases = size(data, 1);
                    poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
                    posprods    = data' * poshidprobs;  % i,j element is vi^0 * hj^0 from Hinton 2006 3.1
                    poshidact   = sum(poshidprobs);
                    posvisact = sum(data);
                    
                    %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    poshidstates = poshidprobs > rand(numcases,numhid);
                    
                    %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
                    neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
                    negprods  = negdata'*neghidprobs;   % i,j element is vi^1 * hj^1 from Hinton 2006 3.1
                    neghidact = sum(neghidprobs);
                    negvisact = sum(negdata);
                    
                    %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    err= sum(sum( (data-negdata).^2 ));
                    errsum = err + errsum;
                    
                    if epoch>5,
                        momentum = self.finalmomentum;
                    else
                        momentum = self.initialmomentum;
                    end;
                    
                    %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % posprods-negprods is gradient of log prob of training data.
                    vishidinc = momentum*vishidinc + ...
                        self.epsilonw*( (posprods-negprods)/numcases - self.weightcost*vishid);
                    visbiasinc = momentum*visbiasinc + (self.epsilonvb/numcases)*(posvisact-negvisact);
                    hidbiasinc = momentum*hidbiasinc + (self.epsilonhb/numcases)*(poshidact-neghidact);
                    
                    vishid = vishid + vishidinc;
                    visbiases = visbiases + visbiasinc;
                    hidbiases = hidbiases + hidbiasinc;
                    
                    %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                end
                fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
            end;
            
            
            % Values (batchposhidprobs) from this layer are used to train the next.
            % Record weights and biases for this layer.
            self.W(1:end-1,:) = vishid;
            self.W(end,:) = hidbiases;
            self.visbias = visbiases;
        end
    end
end


