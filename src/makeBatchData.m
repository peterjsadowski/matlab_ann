function batchdata = makeBatchData(data, batchsize)
% Make a cell array where each element is data struct with .input, .target fields.

if ~isstruct(data)
    temp = data;
    clear data;
    data.input = temp;
end

% Randomize order for batches. We want mix of each class in each batch.
data = randomOrder(data);

% Number of batches is at least one. If more than one batch, we make sure
% each batch has at minimum batchsize examples. Last one may get extra.
nbatch = max(1, floor(size(data.input, 1)/batchsize));

batchdata = cell(nbatch,1);
for j = 1:nbatch
    batchdata{j} = getBatchData(data, batchsize, j);
end
end

function batchdata = getBatchData(data, batchsize, i)
% Get the i-th batch of data. Leftovers are included in last batch.
nfullbatch = floor(size(data.input,1)/batchsize);
if i < nfullbatch
    indices = 1+(i-1)*batchsize:i*batchsize;
elseif i == nfullbatch
    % Include any extra points in the last batch.
    indices = 1+(i-1)*batchsize:size(data.input, 1);
elseif i == 1 && nfullbatch == 0
    indices = 1:size(data.input,1);
else
    error('Index exceeds number of batches.');
end
batchdata.input  = data.input(indices,:);
if isfield(data, 'target')
    batchdata.target = data.target(indices,:);
end
end