function data = randomOrder(data)
% Randomize the order of the data. 

if ~isstruct(data)
    % Data is a simple matrix. Randomize the rows.
    N = size(data, 1);
    order = randperm(N);
    data = data(order, :); % Randomize rows.
    return
else
    % If data is a struct with .input and .target, then randomize those together.
    N = size(data.input,1);
    order = randperm(N);
    data.input = data.input(order,:);
    if isfield(data, 'target')
        assert(size(data.target,1) == N);
        data.target = data.target(order, :);
    end
    
    % Don't bother randomizing test data.
     
    % Randomize unlabeled.
    if isfield(data, 'unlabeled')
        data.unlabeled = randomOrder(data.unlabeled);
    end
end
end