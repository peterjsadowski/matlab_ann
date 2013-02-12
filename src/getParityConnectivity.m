function connectivity =  getParityConnectivity(ninput)

assert(mod(log2(ninput),1) == 0);
nlayers = 2* log2(ninput);
connectivity = cell(1, nlayers);

% % Version 1: Minimum number of units.
% for i = 1:2:nlayers
%    nin  = ninput / (2^floor(i/2));
%    nout = nin;
%    temp = repmat({ones(2)}, 1, nin/2);
%    connectivity{i} =  [blkdiag(temp{:}); ones(1, nout)];
%    
%    nin = nin;
%    nout = nin / 2;
%    temp = repmat({ones(2,1)}, 1, nin/2);
%    connectivity{i+1} =  [blkdiag(temp{:}); ones(1, nout)];
% end

% % Version 2: Hidden layer has 2x number of units necessary
% for i = 1:2:nlayers
%    % Large layer (nout=2*nin). Hidden layer for backprop. 
%    % Each output has 2 inputs + bias term. Each input has to 4 outputs.
%    nin  = ninput / (2^floor(i/2));
%    nout = nin * 2;
%    temp = repmat({ones(2,4)}, 1, nout/4);
%    connectivity{i} =  [blkdiag(temp{:}); ones(1, nout)];
%    
%    % Small layer. Output layer for backprop.
%    % Each output has 4 inputs + bias term. Each input has 1 output.
%    nin = nout;
%    nout = nin / 2 / 2;
%    temp = repmat({ones(4,1)}, 1, nout);
%    connectivity{i+1} =  [blkdiag(temp{:}); ones(1, nout)];
% end

% % Version 3: Hidden layer has 4x number of units necessary
% for i = 1:2:nlayers
%    % Large layer (nout=2*nin). Hidden layer for backprop. 
%    % Each output has 2 inputs + bias term. Each input has to 8 outputs.
%    nin  = ninput / (2^floor(i/2));
%    nout = nin * 4;
%    temp = repmat({ones(2,8)}, 1, nout/8);
%    connectivity{i} =  [blkdiag(temp{:}); ones(1, nout)];
%    
%    % Small layer. Output layer for backprop.
%    % Each output has 8 inputs + bias term. Each input has 1 output.
%    nin = nout;
%    nout = nin / 2 / 4;
%    temp = repmat({ones(8,1)}, 1, nout);
%    connectivity{i+1} =  [blkdiag(temp{:}); ones(1, nout)];
% end

% Version 4: Hidden layer has 4x necessary, output has 2x
nin = ninput;
nout = nin*4;
temp = repmat({ones(2,8)}, 1, nout/8);
connectivity{1} =  [blkdiag(temp{:}); ones(1, nout)];

for i = 2:2:nlayers-1
    % Small layer. Output layer for backprop.
    % Each output has 8 inputs + bias term. Each input has 4 output.
    nin = nout;
    nout = nin / 2;
    temp = repmat({ones(8,4)}, 1, nout/4);
    connectivity{i} =  [blkdiag(temp{:}); ones(1, nout)];
   
   % Large layer (nout=2*nin). Hidden layer for backprop. 
   % Each output has 4 inputs + bias term. Each input has to 4 outputs.
   nin  = nout;
   nout = nin;
   temp = repmat({ones(4,4)}, 1, nout/4);
   connectivity{i+1} =  [blkdiag(temp{:}); ones(1, nout)];
end
nin = nout;
nout = 1;
temp = repmat({ones(nin,1)}, 1, 1);
connectivity{nlayers} =  [blkdiag(temp{:}); ones(1, nout)];

end