function data = load_simulated(nclust, nper, nfeat, p, noisep)
% Create simulated binomial data.
% Each cluster has a mean with some bitwise noise.

% nclust = 10;           % Number of clusters.
% nper   = 100;           % Number of samples per cluster.
% nfeat  = 100; % Number of informative features.

ntotal = nclust * nper; % Number of samples total.

data.input = zeros(ntotal, nfeat);
data.target = zeros(ntotal, nclust);
data.testinput = zeros(ntotal, nfeat);
data.testtarget = zeros(ntotal, nclust);
for i = 1:nclust
    clustcentroid = rand(1, nfeat) < p;
    idx = 1+(i-1)*nper:i*nper;
    
    noise = rand(nper, nfeat) < noisep;
    data.input(idx, :) = xor(repmat(clustcentroid, nper, 1), noise);
    data.target(idx, i) = 1;
    
    noise = rand(nper, nfeat) < noisep;
    data.testinput(idx, :) = xor(repmat(clustcentroid, nper, 1), noise);
    data.testtarget(idx, i) = 1;
end


% % Heatmap
% figure(1)
% colormap(gray)
% imagesc(data)
end