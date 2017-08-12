%% Function Description
% This function trains a deep gait network which incorporates motion
% components, pose components, kinematic model-based joint angle
% components, image components to classify a gait as a threat or non-threat
function dp = deep_gait_networ_initialize(opts_motion_rbm,opts_image_rbm,opts_crbm)

% Initialize the random number generator
rand('state',0);

% using RBM to obtain motion latent vector
dp.rbm_motion.epsilon_w = opts_rbm.epsilon_w;
dp.rbm_motion.epsilon_vb = opts_rbm.epsilon_vb;
dp.rbm_motion.epsilon_vc = opts_rbm.epsilon_vc;
dp.rbm_motion.momentum = opts_rbm.momentum;
dp.rbm_motion.weightcost = opts_rbm.weightcost;

% initializing the weights and weight increments of each rbm
dp.rbm_motion.W = 0.1 * randn(opts_motion_rbm.sizes(2),opts_motion_rbm.sizes(1);
dp.rbm_motion.del_W = zeros(dp.sizes(u+1),dp.sizes(u));

% bias for visible 
dp.rbm_motion.bv = zeros(dp.sizes(u),1);
dp.rbm_motion.del_bv = zeros(dp.sizes(u),1);

% bias for hidden
dp.rbm_motion.bh = zeros(dp.sizes(u+1),1);
dp.rbm_motion.del_bh = zeros(dp.sizes(u+1),1);


dim_X = size(X,2);
dp.sizes = [dim_X , opts_rbm.sizes];
dp.backprop_size = [dp.sizes dp.sizes(end-1:-1:1)]; % size of each layer in stacked auto-encoder

num_of_layers = numel(dp.sizes); % number of layers including the input layer
num_rbms = numel(dp.sizes) - 1;
fprintf('Pretraining the network with %d RBMs\n',num_rbms);

% initialize the corresponding rbms in the deep belief architecture
for u = 1:1:num_rbms-1
    % setting the options for each rbm
    dp.rbm{u}.epsilon_w = opts_rbm.epsilon_w;
    dp.rbm{u}.epsilon_vb = opts_rbm.epsilon_vb;
    dp.rbm{u}.epsilon_vc = opts_rbm.epsilon_vc;
    dp.rbm{u}.momentum = opts_rbm.momentum;
    dp.rbm{u}.weightcost = opts_rbm.weightcost;
    
    % initializing the weights and weight increments of each rbm
    dp.rbm{u}.W = 0.1 * randn(dp.sizes(u+1),dp.sizes(u));
    dp.rbm{u}.del_W = zeros(dp.sizes(u+1),dp.sizes(u));
    
    % bias for visible 
    dp.rbm{u}.bv = zeros(dp.sizes(u),1);
    dp.rbm{u}.del_bv = zeros(dp.sizes(u),1);
    
    % bias for hidden
    dp.rbm{u}.bh = zeros(dp.sizes(u+1),1);
    dp.rbm{u}.del_bh = zeros(dp.sizes(u+1),1);
end

% initialization for the last layer
% setting the options for each rbm
dp.rbm{num_rbms}.epsilon_w = 0.01*opts_rbm.epsilon_w; % rate is 0.001 for the linear layer
dp.rbm{num_rbms}.epsilon_vb = 0.01*opts_rbm.epsilon_vb;
dp.rbm{num_rbms}.epsilon_vc = 0.01*opts_rbm.epsilon_vc;
dp.rbm{num_rbms}.momentum = opts_rbm.momentum;
dp.rbm{num_rbms}.weightcost = opts_rbm.weightcost;

% initializing the weights and weight increments of each rbm
dp.rbm{num_rbms}.W = 0.1 * randn(dp.sizes(num_rbms+1),dp.sizes(num_rbms));
dp.rbm{num_rbms}.del_W = zeros(dp.sizes(num_rbms+1),dp.sizes(num_rbms));

% bias for visible 
dp.rbm{num_rbms}.bv = zeros(dp.sizes(num_rbms),1);
dp.rbm{num_rbms}.del_bv = zeros(dp.sizes(num_rbms),1);

% bias for hidden
dp.rbm{num_rbms}.bh = zeros(dp.sizes(num_rbms+1),1);
dp.rbm{num_rbms}.del_bh = zeros(dp.sizes(num_rbms+1),1);


% for stacked auto-encoder, the last layer is a linear layer and no sigmoid
% function
% training the first layer
X_next = X;
fprintf('\nTraining the %d RBM\n',1);
dp.rbm{1} = rbmtrain_seq_gpu(dp.rbm{1},X_next,opts_rbm,flag_seq);
% training the middle layers
for u = 2:1:num_rbms-1
    X_next = applyRBM(dp.rbm{u-1},X_next);
    fprintf('\nTraining the %d RBM\n',u);
    %X_next = rbmup(sae.rbm{u-1},X_next); % propagating and obtaining the outputs of previous rbm
    dp.rbm{u} = rbmtrain_seq_gpu(dp.rbm{u},X_next,opts_rbm,flag_seq); % TODO : Write up the code for rbmtrain_seq
end

% training the final layer which output if linear
%X_next = rbmup(sae.rbm{n-1},X_next);
X_next = applyRBM(dp.rbm{num_rbms-1},X_next);
fprintf('\nTraining the %d RBM\n',num_rbms);
dp.rbm{num_rbms} = rbmtrain_seq_lin_gpu(dp.rbm{num_rbms},X_next,opts_rbm,flag_seq);

% set the weights of the autoencoder which is a neural network with input and output the same: first set of layers for obtaining
% the lower dimensional manifold
for u = 1:1:numel(dp.rbm)
    dp.W{u} = 0.1*randn(dp.sizes(u+1),dp.sizes(u)+1); % the plus one is for adding the bias term
    dp.W{u} = [dp.rbm{u}.W dp.rbm{u}.bh]; % appending the hidden biases with the weights as the total set of weights in the neural network
end

% setting the weights of the second set of layers
for u = numel(dp.rbm)+1:1:2*numel(dp.rbm)
    v = 2*numel(dp.rbm) - u + 1;
    dp.W{u} = 0.1*randn(dp.sizes(v),dp.sizes(v+1)+1);
    dp.W{u} = [dp.rbm{v}.W' dp.rbm{v}.bv];
end

% Fine-tuning back-propagation training of sae
opts_rbm.numepochs = opts_rbm.numepochs * 3;
fprintf('\nPreTraining complete\n');
fprintf('Training the stacked auto-encoder as a neural network\n');
dp = sae_nn_train_gpu(dp,X,opts_rbm,flag_seq);
fprintf('Training of SAE complete\n');

end
