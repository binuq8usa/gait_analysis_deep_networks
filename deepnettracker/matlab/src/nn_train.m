% function to compute the stacked autoencoder for a set of features
% Features : cell array containing actions of a class
% This partially used the Deep Learning toolbox in matlab
% sae : stacked auto_encoder structure which has rbms pre-trained structure
% for each layer and a set of weights corresponding to back-propagation nn
function st = nn_train(X,Y,opts,flag_seq)

% write up a deep belief network architecture for pre-training
% write up a common deep belief architecture
rand('state',0);
dim_X = size(X,2);
dim_Y = size(Y,2);
st.sizes = [dim_X , opts.sizes , dim_Y];

num_of_layers = numel(st.sizes); % number of layers including the input layer
num_rbms = num_of_layers - 2; % the final layer will not be trained as an RBM
fprintf('Pretraining the network with %d RBMs\n',num_rbms);

% initialize the corresponding rbms in the deep belief architecture
for u = 1:1:num_rbms
    % setting the options for each rbm
    st.rbm{u}.epsilon_w = opts.epsilon_w;
    st.rbm{u}.epsilon_vb = opts.epsilon_vb;
    st.rbm{u}.epsilon_vc = opts.epsilon_vc;
    st.rbm{u}.momentum = opts.momentum;
    st.rbm{u}.weightcost = opts.weightcost;
    
    % initializing the weights and weight increments of each rbm
    st.rbm{u}.W = 0.1 * randn(st.sizes(u+1),st.sizes(u));
    st.rbm{u}.del_W = zeros(st.sizes(u+1),st.sizes(u));
    
    % bias for visible 
    st.rbm{u}.bv = zeros(st.sizes(u),1);
    st.rbm{u}.del_bv = zeros(st.sizes(u),1);
    
    % bias for hidden
    st.rbm{u}.bh = zeros(st.sizes(u+1),1);
    st.rbm{u}.del_bh = zeros(st.sizes(u+1),1);
end

% for stacked auto-encoder, the last layer is a linear layer and no sigmoid
% function
% training the first layer
X_next = X;
fprintf('\nTraining the %d RBM\n',1);
st.rbm{1} = rbmtrain_seq(st.rbm{1},X_next,opts,flag_seq);
% training the middle layers
for u = 2:1:num_rbms
    X_next = applyRBM(st.rbm{u-1},X_next);
    fprintf('\nTraining the %d RBM\n',u);
    %X_next = rbmup(sae.rbm{u-1},X_next); % propagating and obtaining the outputs of previous rbm
    st.rbm{u} = rbmtrain_seq(st.rbm{u},X_next,opts,flag_seq); % TODO : Write up the code for rbmtrain_seq
end

% set the weights of the autoencoder which is a neural network with input and output the same: first set of layers for obtaining
% the lower dimensional manifold
for u = 1:1:numel(st.rbm)
    st.W{u} = 0.1*randn(st.sizes(u+1),st.sizes(u)+1); % the plus one is for adding the bias term
    st.W{u} = [st.rbm{u}.W st.rbm{u}.bh]; % appending the hidden biases with the weights as the total set of weights in the neural network
end

% setting the weights of the output layer (for regression or
% classification)
st.W{num_rbms+1} = 0.1*randn(st.sizes(num_of_layers),st.sizes(num_of_layers-1)+1);

% % setting the weights of the second set of layers
% for u = numel(st.rbm)+1:1:2*numel(st.rbm)
%     v = 2*numel(st.rbm) - u + 1;
%     st.W{u} = 0.1*randn(st.sizes(v),st.sizes(v+1)+1);
%     st.W{u} = [st.rbm{v}.W' st.rbm{v}.bv];
% end

% Fine-tuning back-propagation training of sae
opts.numepochs = opts.numepochs * 10;
fprintf('\nPreTraining complete\n');
fprintf('Training of a deep neural network\n');

% calling the function approximator or classification function for
% error-backpropagation
st = func_ap_nn_train(st,X,Y,opts,flag_seq);

% st = sae_nn_train(st,X,opts,flag_seq);
fprintf('Training of the deep neural network complete\n');

end
