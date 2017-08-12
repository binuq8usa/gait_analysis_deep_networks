%% Function Description
% This function trains a deep gait network which incorporates motion
% components, pose components, kinematic model-based joint angle
% components, image components to classify a gait as a threat or non-threat
function dp = deep_gait_regression_pretrain(motionVector, kinematicModelVector, imageVector,opts_motion_rbm,opts_image_rbm,flag_whiten)

%% Initialization PHASE
% Initialize the random number generator
rand('state',0);
num_samples = size(motionVector,1);
numDimMotion = size(motionVector,2);
numDimImage = size(imageVector,2);

% Whitening the data 
if(flag_whiten)
    [dp.motionVectorWhite, dp.numRedDimMotion, dp.U_motion, dp.S_motion, dp.V_motion, dp.motionVectorMean, dp.motionVector] = pcaWhiten(motionVector,1,numDimMotion/4);
    [dp.imageVectorWhite, dp.numRedDimImage, dp.U_image, dp.S_image, dp.V_image, dp.imageVectorMean, dp.imageVector] = pcaWhiten(imageVector,1,numDimImage/4);
    
    % normalize the PCA whitened data to range (0,1) for DBN pre-training
    dp.motionVectorMax = max(dp.motionVector(:));
    dp.motionVectorMin = min(dp.motionVector(:));

    dp.imageVectorMax = max(dp.imageVector(:));
    dp.imageVectorMin = min(dp.imageVector(:));

    dp.motionVectorNorm = (dp.motionVector - dp.motionVectorMin)./(dp.motionVectorMax - dp.motionVectorMin);
    dp.imageVectorNorm = (dp.imageVector - dp.imageVectorMin)./(dp.imageVectorMax - dp.imageVectorMin);
    
else % or reduce in size to 32 x 32 (downsample)
    %dp.motionVector = motionVector;
    %dp.imageVector = imageVector;
%     dp.motionVector = zeros(num_samples,numDimMotion/2);
%     dp.imageVector = zeros(num_samples,numDimMotion/2);
%     
%     for kk = 1:1:num_samples
%         dp.motionVector(kk,:) = imresize(motionVector(kk,:),0.5,'nearest');
%         dp.imageVector(kk,:) = imresize(imageVector(kk,:),0.5,'nearest');
%     end
    
    dp.motionVectorNorm = motionVector;
    dp.imageVectorNorm = imageVector;
    
    dp.numRedDimMotion = numDimMotion;
    dp.numRedDimImage = numDimImage;
end




% % using Motion RBM to obtain motion latent vector
% % TODO: Use Gaussian RBM
dim_motion = size(motionVector,2);
dp.motion_rbm_sizes = [dim_motion , opts_motion_rbm.sizes];
num_motion_rbms = length(dp.motion_rbm_sizes)-1;

% Pretrain the first layer of the RBM for u and v vector seperately
% motionVector_U = motionVector(:,1:dim_motion/2);
% motionVector_V = motionVector(:,dim_motion/2+1:end);

% R_u = default_rbm(dim_motion/2,opts_motion_rbm.sizes(1));
% R_v = default_rbm(dim_motion/2,opts_motion_rbm.sizes(1));
% 
% R_u = train_rbm(R_u,motionVector_U);
% R_v = train_rbm(R_v,motionVector_V);


% Seperating it out is not a good idea as the error keeps fluctuating
% R_motionU_dbm = default_dbm([dim_motion/2 opts_motion_rbm.sizes]);
% R_motionU_dbm = dbm(R_motionU_dbm,motionVector_U);
% 
% R_motionV_dbm = default_dbm([dim_motion/2 opts_motion_rbm.sizes]);
% R_motionV_dbm = dbm(R_motionV_dbm,motionVector_V);

% Try directly using Deep Belief Nets
R_motion_dbn = default_dbn([dp.numRedDimMotion opts_motion_rbm.sizes]);
%R_motion_dbn.learning.persistent_cd =0; 
R_motion_dbn.iteration.n_epochs = opts_motion_rbm.numepochs;
%R_motion_dbn.learning.weight_decay = opts_motion_rbm.weightcost; R_motion_dbn.learning.momentum = opts_motion_rbm.momentum;
%R_motion_dbn.learning.lrate = opts_motion_rbm.epsilon_w;
R_motion_dbn = dbn(R_motion_dbn,dp.motionVectorNorm);

% Try using Deep Boltzmann Machines ( This gives less error and better)
% R_motion_dbm = default_dbm([dim_motion opts_motion_rbm.sizes]);
% R_motion_dbm = dbm(R_motion_dbm,motionVector);

% Try MLP with Deep Boltzmann Machines

% Try Stacked de-noising autoencoders

% R_motion_sdae = default_sdae([dim_motion opts_motion_rbm.sizes]);i
% R_motion_sdae.hidden.use_tanh = 1;
% R_motion_sdae = sdae(R_motion_sdae,motionVector);


% using Image RBM to obtain kinematic latent vector
% TODO: Use Gaussian RBM
%kinematicVector = imageVector; % combining both joint model and pose model vector
dim_image = size(imageVector,2);
dp.image_rbm_sizes = [dp.numRedDimImage , opts_image_rbm.sizes];
num_image_rbms = length(dp.image_rbm_sizes)-1;

% R_kinematic_dbm = default_dbm([dim_kinematic opts_kinematic_rbm.sizes]);
% R_kinematic_dbm = dbm(R_kinematic_dbm,kinematicVector);

R_image_dbn = default_dbn([dp.numRedDimImage opts_image_rbm.sizes]);
%R_image_dbn.learning.persistent_cd = 0; 
R_image_dbn.iteration.n_epochs = opts_image_rbm.numepochs;
%R_image_dbn.learning.weight_decay = opts_image_rbm.weightcost; R_image_dbn.learning.momentum = opts_image_rbm.momentum;
%R_image_dbn.learning.lrate = opts_image_rbm.epsilon_w;
R_image_dbn = dbn(R_image_dbn,dp.imageVectorNorm);

dp.R_motion_dbn = R_motion_dbn;
dp.R_image_dbn = R_image_dbn;

% get the data forwarded through the trained DBN layer

[motionLatentVector] = dbn_forward(dp.motionVectorNorm, R_motion_dbn);
[imageLatentVector] = dbn_forward(dp.imageVectorNorm, R_image_dbn);

g=gpuDevice;
reset(g);

% train the MLP network for regression
% Two types of training
% One is standard units, the other is reLu units
% The one with ReLU units, pretrained with DBM
% model 1
%mlp_layers{1} = [dp.motion_rbm_sizes(end)+dp.image_rbm_sizes(end) 2048 256 size(kinematicModelVector,2)];
%mlp_layers{2} = [dp.motion_rbm_sizes(end)+dp.image_rbm_sizes(end) 256 64 size(kinematicModelVector,2)];
mlp_layers{1} = [dp.motion_rbm_sizes(end)+dp.image_rbm_sizes(end) 256 64 size(kinematicModelVector,2)];
ind = crossvalind('Kfold',size(motionLatentVector,1));

% Do Cross Validation
% Plot the error from the validation (last element) and from the
% recognition error (last element) for part of the layer.
% Do we average the errors over the validation set for a particular model?
% Yes we do. Change the model to different configurations (number of
% layers, number of units)
% keeping a set for cross validation so that model is better trained

for mm = 1:1:1
    ll = unique(ind);
    for k = ll(1)
        % cross validate the model
        dp.mlp_regression_network_model{mm,k} = default_mlp(mlp_layers{mm});
        dp.mlp_regression_network_model{mm,k}.output.binary = 0;
        dp.mlp_regression_network_model{mm,k}.hidden.use_tanh = 2;
        dp.mlp_regression_network_model{mm,k}.iteration.n_epochs = 2500;
        dp.mlp_regression_network_model{mm,k}.learning.lrate = 0.001;
        
        validInd = find(ind == k); % find the validation indices
        trainInd = find(ind ~= k); % find the training indices
        
        patches = [motionLatentVector(trainInd,:) imageLatentVector(trainInd,:)];
        targets = kinematicModelVector(trainInd,:);
        valid_patches = [motionLatentVector(validInd,:) imageLatentVector(validInd,:)];
        valid_targets = kinematicModelVector(validInd,:);
        valid_portion = 1;
        cvp = 0;
        dp.mlp_regression_network_model{mm,k} = mlp_custom(dp.mlp_regression_network_model{mm,k},patches, targets, valid_patches, valid_targets, valid_portion,cvp );
        
    end
end

dp.mlp_regression_network = dp.mlp_regression_network_model{mm,k};

% 
% % train the MLP network
% mm = 1;
% dp.mlp_regression_network = default_mlp(mlp_layers{mm});
% dp.mlp_regression_network.output.binary = 0;
% dp.mlp_regression_network.hidden.use_tanh = 2;
% dp.mlp_regression_network.iteration.n_epochs = 2500;
% dp.mlp_regression_network.learning.lrate = 0.001;
% 
% patches = [motionLatentVector imageLatentVector];
% targets = kinematicModelVector;
% %valid_patches = [motionLatentVector imageLatentVector];
% %valid_targets = kinematicModelVector;
% %valid_portion = 1;
% cvp = 0;
% dp.mlp_regression_network = mlp_custom(dp.mlp_regression_network,patches, targets);

dp.name = 'GaitNetwork';


end
