function printresults(dataname, varargin)
path('/home/pjsadows/ml/prototyping/src/', path); 

if nargin > 1
    pattern = varargin{1};
    %fprintf('Finding only filenames that include pattern: %s\n', pattern');
end

filepath = ['results/', dataname, '/'];
%filepath = 'results/jan8_processed_highlevel7/';
listing = dir(strcat(filepath, '/*.mat'));

for i = 1:length(listing)
    filename = listing(i).name;
    fullpath = fullfile(filepath, listing(i).name);
    
    if exist('pattern','var') && isempty(strfind(filename, pattern))
        % Pattern specified and not found.
        continue
    end
    
    import = load(fullpath);
    n = import.n;
    
%     n.ErrorTrainClass(end);
%     fprintf('%s ErrorTrain\n', auc, listing(i).name)
    
    fprintf('%s\t', filename);
    n.printstatus % Will print a line with error info.
end

end