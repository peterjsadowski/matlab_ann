
% Grab text data.
% import file....
Test = data;

for i = 1:size(textdata, 1)
    list = regexp(textdata{i,4}, ' ', 'split');
    num = str2double(list(1));
    Train(i) = num;    
end


%% Plot

nepoch = length(Train);
assert(length(Test) == nepoch);

figure(1)
clf
hold on
plot(Train, 'k-');
plot(Test, 'b-');
hold off
xlabel('Epoch')
ylabel('Classification Error')
xlim([0,nepoch])
ylim([0,1])
names = {'DT Train', 'DT Test'};
legend(names)

fn = 'results/tempresults.eps';
saveas(gcf, fn, 'eps')