
data = loaddata('test');
p = perceptron(100, 10, 'logsig', 'CrossEntropyError');
p = train_gd(p, data, 10);
imagesc(apply(p, data.input))

p = perceptron(100, 10, 'logsig', 'CrossEntropyError');
p = initialize(p, 10);
p = train_gd(p, data, 10);
imagesc(apply(p, data.input))
imagesc(p.W)
            