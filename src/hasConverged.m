function converged = hasConverged(errors, varargin)
% Return true if error meets convergence criteria.


p = inputParser;
addRequired(p, 'errors', @isnumeric);
addOptional(p, 'minsteps', 2, @isnumeric);
addOptional(p, 'window',   1, @isnumeric);
addOptional(p, 'threshold',0, @isnumeric);
parse(p, errors, varargin{:});

errors = p.Results.errors;
minsteps = p.Results.minsteps;
window = p.Results.window;
threshold = p.Results.threshold;

if (length(errors) >= minsteps-1) && (length(errors) > window)
    delta = errors(end) - errors(end-window);
    if delta >= - threshold
        converged = true;
        return
    end
end
converged = false;
end
