%% Function Description
% This function trains a deep gait network which incorporates motion
% components, pose components, kinematic model-based joint angle
% components, image components to classify a gait as a threat or non-threat
function dp = deep_gait_network_pretrain(motionVector, kinematicModelVector, imageVector,opts_motion_rbm,opts_kinematic_rbm,opts_combined_rbm)

%% Initialization PHASE
% Initialize the random number generator
rand('state',0);

% % using Motion RBM to obtain motion latent vector
% % TODO: Use Gaussian RBM
dim_motion = size(motionVector,2);
dp.motion_rbm_sizes = [dim_motion , opts_motion_rbm.sizes];
num_motion_rbms = length(dp.motion_rbm_sizes)-1;
for u = 1:1:num_motion_rbms
    dp.rbm_motion{u}.epsilon_w = opts_motion_rbm.epsilon_w;
    dp.rbm_motion{u}.epsilon_vb = opts_motion_rbm.epsilon_vb;
    dp.rbm_motion{u}.epsilon_vc = opts_motion_rbm.epsilon_vc;
    dp.rbm_motion{u}.momentum = opts_motion_rbm.momentum;
    dp.rbm_motion{u}.weightcost = opts_motion_rbm.weightcost;

    % initializing the weights and weight increments of each rbm
    dp.rbm_motion{u}.W = 0.1 * randn(dp.motion_rbm_sizes(u+1),dp.motion_rbm_sizes(u));
    dp.rbm_motion{u}.del_W = zeros(dp.motion_rbm_sizes(u+1),dp.motion_rbm_sizes(u));

    % bias for visible 
    dp.rbm_motion{u}.bv = zeros(dp.motion_rbm_sizes(u),1);
    dp.rbm_motion{u}.del_bv = zeros(dp.motion_rbm_sizes(u),1);

    % bias for hidden
    dp.rbm_motion{u}.bh = zeros(dp.motion_rbm_sizes(u+1),1);
    dp.rbm_motion{u}.del_bh = zeros(dp.motion_rbm_sizes(u+1),1);
end

% training the first layer of motion RBM
X_next = motionVector;
fprintf('\nTraining the %d Layer of Motion RBM\n',1);
dp.rbm_motion{1} = rbmtrain_seq_gpu(dp.rbm_motion{1},X_next,opts_motion_rbm,0);
% training the middle layers
for u = 2:1:num_motion_rbms
    X_next = applyRBM(dp.rbm_motion{u-1},X_next);
    fprintf('\nTraining the %d Layer of Motion RBM\n',u);
    %X_next = rbmup(sae.rbm{u-1},X_next); % propagating and obtaining the outputs of previous rbm
    dp.rbm_motion{u} = rbmtrain_seq_gpu(dp.rbm_motion{u},X_next,opts_motion_rbm,0); % TODO : Write up the code for rbmtrain_seq
end

motionLatentVector = applyRBM(dp.rbm_motion{num_motion_rbms},X_next);
fprintf('\n** Completed training of Motion RBM\n');

% using Image RBM to obtain kinematic latent vector
% TODO: Use Gaussian RBM
kinematicVector = [kinematicModelVector, imageVector]; % combining both joint model and pose model vector
dim_kinematic = size(kinematicVector,2);
dp.kinematic_rbm_sizes = [dim_kinematic , opts_kinematic_rbm.sizes];
num_kinematic_rbms = length(dp.kinematic_rbm_sizes)-1;
for u = 1:1:num_kinematic_rbms
    dp.rbm_kinematic{u}.epsilon_w = opts_kinematic_rbm.epsilon_w;
    dp.rbm_kinematic{u}.epsilon_vb = opts_kinematic_rbm.epsilon_vb;
    dp.rbm_kinematic{u}.epsilon_vc = opts_kinematic_rbm.epsilon_vc;
    dp.rbm_kinematic{u}.momentum = opts_kinematic_rbm.momentum;
    dp.rbm_kinematic{u}.weightcost = opts_kinematic_rbm.weightcost;

    % initializing the weights and weight increments of each rbm
    dp.rbm_kinematic{u}.W = 0.1 * randn(dp.kinematic_rbm_sizes(u+1),dp.kinematic_rbm_sizes(u));
    dp.rbm_kinematic{u}.del_W = zeros(dp.kinematic_rbm_sizes(u+1),dp.kinematic_rbm_sizes(u));

    % bias for visible 
    dp.rbm_kinematic{u}.bv = zeros(dp.kinematic_rbm_sizes(u),1);
    dp.rbm_kinematic{u}.del_bv = zeros(dp.kinematic_rbm_sizes(u),1);

    % bias for hidden
    dp.rbm_kinematic{u}.bh = zeros(dp.kinematic_rbm_sizes(u+1),1);
    dp.rbm_kinematic{u}.del_bh = zeros(dp.kinematic_rbm_sizes(u+1),1);
end

% training the first layer of kinematic RBM
X_next = kinematicVector;
fprintf('\nTraining the %d Layer of kinematic RBM\n',1);
dp.rbm_kinematic{1} = rbmtrain_seq_gpu(dp.rbm_kinematic{1},X_next,opts_kinematic_rbm,0);
% training the middle layers
for u = 2:1:num_kinematic_rbms
    X_next = applyRBM(dp.rbm_kinematic{u-1},X_next);
    fprintf('\nTraining the %d Layer of kinematic RBM\n',u);
    %X_next = rbmup(sae.rbm{u-1},X_next); % propagating and obtaining the outputs of previous rbm
    dp.rbm_kinematic{u} = rbmtrain_seq_gpu(dp.rbm_kinematic{u},X_next,opts_kinematic_rbm,0); % TODO : Write up the code for rbmtrain_seq
end

kinematicLatentVector = applyRBM(dp.rbm_kinematic{num_kinematic_rbms},X_next);
fprintf('\n** Completed training of Kinematic RBM\n');
% dp.rbm_kinematic{num_kinematic_rbms} = rbmtrain_seq_lin_gpu(dp.rbm_kinematic{num_kinematic_rbms},X_next,opts_kinematic_rbm,flag_seq);

% % initialization for the last layer which should be a linear output
% % % setting the options for each rbm
% 
% combinedVector = [motionLatentVector, kinematicLatentVector];
% dim_combined = size(combinedVector,2);
% dp.combined_rbm_sizes = [dim_combined , opts_combined_rbm.sizes]; % assuming that opts_combined_rbm.sizes is a single element.
% num_combined_rbms = 1;
% 
% dp.rbm_combined{num_combined_rbms}.epsilon_w = 0.01*opts_combined_rbm.epsilon_w; % rate is 0.001 for the linear layer
% dp.rbm_combined{num_combined_rbms}.epsilon_vb = 0.01*opts_combined_rbm.epsilon_vb;
% dp.rbm_combined{num_combined_rbms}.epsilon_vc = 0.01*opts_combined_rbm.epsilon_vc;
% dp.rbm_combined{num_combined_rbms}.momentum = opts_combined_rbm.momentum;
% dp.rbm_combined{num_combined_rbms}.weightcost = opts_combined_rbm.weightcost;
% dp.rbm_combined{num_combined_rbms}.W = 0.1 * randn(dp.sizes(num_combined_rbms+1),dp.sizes(num_combined_rbms));
% dp.rbm_combined{num_combined_rbms}.del_W = zeros(dp.sizes(num_combined_rbms+1),dp.sizes(num_combined_rbms));
% dp.rbm_combined{num_combined_rbms}.bv = zeros(dp.sizes(num_combined_rbms),1);
% dp.rbm_combined{num_combined_rbms}.del_bv = zeros(dp.sizes(num_combined_rbms),1);
% dp.rbm_combined{num_combined_rbms}.bh = zeros(dp.sizes(num_combined_rbms+1),1);
% dp.rbm_combined{num_combined_rbms}.del_bh = zeros(dp.sizes(num_combined_rbms+1),1);
% 
% X_next = combinedVector;
% dp.rbm_combined{num_combined_rbms} = rbmtrain_seq_lin_gpu(dp.rbm_combined{num_combined_rbms},X_next,opts_combined_rbm,0);


end
