function str = arch2str(arch)
if all(arch == arch(1))
    str = sprintf('%dx%d', arch(1), length(arch));
elseif length(arch) < 10
    % Show all x in the form n0_n1_n2_..._nx
    str = sprintf('_%d', arch);
    str = str(2:end); % Remove leading underscore.
else
    % Show first x then _etc.
    str = sprintf('_%d', arch(1:10));
    str = [str(2:end), '_etc']; % Remove leading underscore
end
end