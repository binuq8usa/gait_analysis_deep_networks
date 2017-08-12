function [xOut] = dbn_forward(x0, D)

layers = D.structure.layers;
n_layers = length(layers);

% initial sample
vh = x0;
for l = 2:n_layers-1
    vh = sigmoid(bsxfun(@plus, vh * D.rec.W{l-1}, D.rec.biases{l}'));
    vh = binornd(1, vh);
end

vh = sigmoid(bsxfun(@plus, vh * D.top.W, D.top.hbias'));
vh = binornd(1, vh);

xOut = vh;

