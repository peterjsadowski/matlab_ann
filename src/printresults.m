function printresults(resultdir)
% Function for reading and printing results,
% which are stored in indivual files under the resultdir directory.

% List of input files.
filelist = dir(strcat(resultdir, '/*.mat'));

% warning('off',id)
% Read all data.
for i = 1:length(filelist);
   load(fullfile(resultdir, filelist(i).name));
   
   
   if n.fullarch(end) == 784
       % MNIST auto
       
       %fprintf('%s %f\n', param2str(n), n.ErrorTrainClass(end));
   end
   
   if n.fullarch(end) == 10 && n.fullarch(1) == 784
       % MNIST classify
       if n.fullarch(2) == 100 
           fprintf('%s %f\n', param2str(n), n.ErrorTrainClass(end));
       end
   else
       
       continue
   end
   
end

        
% Output files.
% outputnetgd(results, resultdir);
%outputall(results, resultdir);

end

function outputall(results, resultdir)
% Output files.
fid       = fopen(strcat(resultdir, '/errors.csv'), 'w');
fid_class = fopen(strcat(resultdir, '/classerrors.csv'), 'w');
fid_times = fopen(strcat(resultdir, '/times.csv'), 'w');

error     = [];
time      = [];
classerror= [];

nmethod = 3;

datanames = fields(results);
for idata = 1:length(datanames)
    dataname = datanames{idata};
    archstrs = fields(results.(dataname));
    for iarch = 1:length(archstrs)
        archstr = archstrs{iarch};
        methods = fields(results.(dataname).(archstr));
        
        % Create a new row of our results matrix.
        i = size(error, 1) + 1;
        error(end+1,:)      = inf * ones(1, nmethod);
        time(end+1,:)       = inf * ones(1, nmethod);
        classerror(end+1,:) = inf * ones(1, nmethod);
        
        % Read all numbers for this row.
        
        for imeth = 1:length(methods)
            method = methods{imeth};
            switch method
                case {'net', 'new'}
                    mid = 1;
                case 'gd'
                    mid = 2;
                case 'hebb'
                    mid = 3;
            end
            % Fill in result matrix.
            error(end, mid)      = results.(dataname).(archstr).(method).error;
            time(end, mid)       = results.(dataname).(archstr).(method).time;
            classerror(end, mid) = results.(dataname).(archstr).(method).classerror;
        end
        
        % Headers
        if idata == 1 && iarch == 1
            fprintf('\n\nNET\tGD\tHEBB\n');
            fprintf(fid, 'Dataset,Hidden encoding layers,NET,GD,HEBB\n');
            fprintf(fid_times, 'Dataset,Hidden encoding layers,NET,GD,HEBB\n');
            fprintf(fid_class, 'Dataset,Hidden encoding layers,NET,GD,HEBB\n');
        end
        
        % Format archstr.
        archstrnice = strrep(archstr, 'layers_', ''); % Remove prefix.
        archstrnice = strrep(archstrnice, '_', '\_');     % For latex.
        
        % To screen.
        fprintf('%s%s\t%s\n', sprintf('%0.3f\t', classerror(i,:)), dataname, archstrnice);
        % To file.
        %    fprintf(fid, '%s\t%d\t%d\t%s\n', name, error(i,1), error(i,2), error(i,3));
        fprintf(fid, '%s,%s%s\n', dataname, archstrnice, sprintf(',%0.3f', error(i,:)));
        fprintf(fid_times, '%s,%s%s\n', dataname, archstrnice, sprintf(',%0.0f', time(i,:)));
        fprintf(fid_class, '%s,%s%s\n', dataname, archstrnice, sprintf(',%0.3f', classerror(i,:)));
    end
end
fclose(fid);
fclose(fid_class);
fclose(fid_times);
end


function outputnetgd(results, resultdir)
% Output files.
fid       = fopen(strcat(resultdir, '/errors.csv'), 'w');
fid_times = fopen(strcat(resultdir, '/times.csv'), 'w');
fid_class = fopen(strcat(resultdir, '/classerrors.csv'), 'w');

% Headers
fprintf('\n\nNET\tGD\n');
fprintf(fid, 'Dataset,Hidden encoding layers,NET,GD\n');
fprintf(fid_times, 'Dataset,Hidden encoding layers,NET,GD\n');
fprintf(fid_class, 'Dataset,Hidden encoding layers,NET,GD\n');

error     = inf *ones(0, 2);
time      = inf *ones(0, 2);
classerror= inf *ones(0, 2);

datanames = fields(results);
for idata = 1:length(datanames)
    dataname = datanames{idata};
    archstrs = fields(results.(dataname));
    for iarch = 1:length(archstrs)
        archstr = archstrs{iarch};
        methods = fields(results.(dataname).(archstr));
        
        % Create a new row of our results matrix.
        i = size(error, 1) + 1;
        error(i,:)      = inf * ones(1, 2);
        time(i,:)       = inf * ones(1, 2);
        classerror(i,:) = inf * ones(1, 2);
        % Get method index for results matrix.
        for imeth = 1:length(methods)
            method = methods{imeth};
            switch method
                case {'net', 'new'}
                    mid = 1;
                case 'gd'
                    mid = 2;
            end
            % Fill in result matrix.
            error(end, mid)      = results.(dataname).(archstr).(method).error;
            time(end, mid)       = results.(dataname).(archstr).(method).time;
            classerror(end, mid) = results.(dataname).(archstr).(method).classerror;
        end
        % Format archstr.
        archstrnice = strrep(archstr, 'layers_', ''); % Remove prefix.
        archstrnice = strrep(archstrnice, '_', '\_');     % For latex.
   
        % To screen.
        fprintf('%0.3f\t%0.3f\t%s\t%s\n', classerror(i,1), classerror(i,2), dataname, archstrnice);
        % To file.
        %    fprintf(fid, '%s\t%d\t%d\t%s\n', name, error(i,1), error(i,2), error(i,3));
        fprintf(fid, '%s,%s,%0.3f,%0.3f\n', dataname, archstrnice, error(i,1), error(i,2));
        fprintf(fid_times, '%s,%s,%0.0f,%0.0f\n', dataname, archstrnice, time(i,1), time(i,2));
        fprintf(fid_class, '%s,%s,%0.3f,%0.3f\n', dataname, archstrnice, classerror(i,1), classerror(i,2));
    end
end
fclose(fid);
fclose(fid_times);
end

