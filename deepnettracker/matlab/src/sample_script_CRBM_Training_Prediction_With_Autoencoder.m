% sample CRBM Training wiht autoencoder vectors
a_crbm_train_inputs = [];
Size = [];
for ii=1:1:length(data_angles_seq_train)
    % get the training sequence
    train_seq_jangles = data_angles_seq_train{ii}{1};
    train_seq_pos = data_angles_seq_train{ii}{2};
    class_vector = data_angles_seq_train{ii}{3};
    
    train_seq_motion_vector = data_angles_seq_train{ii}{4};
    
    num_frames = size(train_seq_pos,1);
    
    pose_vector = [train_seq_pos train_seq_jangles];
    
    % accumulate the input data
    %a_crbm_train_inputs = [a_crbm_train_inputs ; [train_seq_pos train_seq_jangles]];
    
    % TODO: Need to combine motion vector with joint trajectory
    % Not the shape descriptro
    motionVector = train_seq_motion_vector(1:num_frames,:);
    
    [X_out, motionLatentVector] = sae_nn_ff(st,motionVector);
    
    % tensor product of pose vector and joint angles
    P_desc = [];
    for jj = 1:1:num_frames
        P_desc = [P_desc ; kron(pose_vector(jj,:),motionLatentVector(jj,:))];
    end
    
    a_crbm_train_inputs = [a_crbm_train_inputs ; P_desc];
    %Size = [Size ; num_frames];
    Size = [Size ; size(P_desc,1)];
    %labels_train = [labels_train; class_vector]; % one label from sequence
    %labels_train = [labels_train; class_vector(1)*ones(size(P_desc,1),1)];
end

% setting the size
opts_crbm.seq_lengths = Size;
%X_gpu = gpuArray(a_crbm_train_inputs);
crbm_gpu = trainCRBM_gpu(a_crbm_train_inputs,opts_crbm);