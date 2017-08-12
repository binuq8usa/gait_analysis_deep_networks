function [y_out,y_in] = rectifier_nn_ff(st,X)
% This function gets a set of inputs X computed the reconstructed output
% X_out and its internal latent representation

num_images = size(X,1);
num_rbms = numel(st.rbm); % number of layers considered as rbms

%%% Encoder part %%%%
X_next = X;

% sigmoid activation function 
for u = 1:1:num_rbms
    data = [X_next ones(num_images,1)];
    w = st.W{u}'; % getting the layer u weights
    
    % obtaining the output of layer u
    %data_a = 1./(1 + exp(-1*(data*w)));
    
    x = data * w;
    data_a = max(x,0); % rectilinear units
    %x_norm = (x - repmat(mean(x,1),num_images,1))./std(
    
    % serving as input to next layer
    X_next = data_a;
    
end

y_in = X_next; % input to the last pair of layers ( output (non rbm) layer)
% last layer which is linear ; since this is a function approximator
data = [X_next ones(num_images,1)];
w = st.W{num_rbms+1}';
%data_a = data*w;
%data_a = 1./(1 + exp(-1*(data*w)));
data_a = data * w; % output is linear
y_out = data_a;

end

