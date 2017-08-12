%% Script to train a deep neural network for function approximation
% set the parameters
% TODO: Try using rectilinear units in autoencoder training and backpropagation
N_sim = 1;
opts_crbm.n{1} = 10;
opts_crbm.n{2} = 10;
opts_crbm.sizes = [128 128]; % originally 150
opts_crbm.gsd = 1;
opts_crbm.nt = 5; % number of previous frames used for computing weights in the current frame
opts_crbm.batchsize = 100;
opts_crbm.epsilon_w = 1e-5; % changed to 10^-4 because 10^-3 wasn't converging
opts_crbm.epsilon_bi = 1e-5;
opts_crbm.epsilon_bj = 1e-5;
opts_crbm.epsilon_A = 1e-5;
opts_crbm.epsilon_B = 1e-5;
opts_crbm.w_decay = 0.0002;
opts_crbm.momentum = 0.9;
opts_crbm.num_epochs = 2000; % 2000 iterations not required as error is same at 500 iterations
opts_crbm.numGibbs = 200;
opts_crbm.dropRate = 1;
win_size = 15;

opts_auto.epsilon_w = 0.15;
opts_auto.epsilon_vb = 0.15;
opts_auto.epsilon_vc = 0.15;
opts_auto.momentum = 0.9;
opts_auto.weightcost = 0.0001;
opts_auto.numepochs = 2000;
opts_auto.sizes = [1024 256]; %  layers divided by 4
opts_auto.batch_size = 100;

% Using precomputed kernel
C_val = 0.5; % Tried different values of C = 0.5, 1, 1.5
opts = sprintf('-s 0 -t 4 -c %d -b 1 -q',C_val);

% load the data_angles_per_seq
data_angles_seq = data_angles_per_seq(:);
data_angles_seq_nonempty = {};
class_labels_seq = [];
num_count = 0;

% Split the train and test of sequences
% Find the class labels for each sequence
for ii=1:1:length(data_angles_per_seq(:))
    if(isempty(data_angles_seq{ii}))
        continue;
    end
    num_count = num_count + 1;
    data_angles_seq_nonempty{num_count} = data_angles_seq{ii};
    % get the labels for each frame of the sequence
    class_vector = data_angles_seq{ii}{3};
    class_labels_seq = [class_labels_seq; class_vector(1)]; % one label from sequence
end

acc = zeros(N_sim,1);
acc2_org = zeros(N_sim,1);
acc2_l1 = zeros(N_sim,1);
acc2_l2 = zeros(N_sim,1);
Train_idx = cell(N_sim,1);
for ns = 1:1:N_sim
    class1_idxs = find(class_labels_seq == 1);
    class2_idxs = find(class_labels_seq == 2);
    
    % SPLIT THE TRAINING/TESTING FROM EACH CLASS : NO UNEVEN DISTRIBUTION OF TRAIN VECTORS
    train_idx1 = randsample(class1_idxs,ceil(2/3 * length(class1_idxs)));
    train_idx2 = randsample(class2_idxs,ceil(2/3 * length(class2_idxs)));
    train_idx = [train_idx1 ; train_idx2];
    Train_idx{ns} = train_idx;
end
%%
st_all = cell(N_sim,1);
crbm_all = cell(N_sim,1);
%for ns = 1:1:N_sim
    ns = 1;
    g=gpuDevice;
    reset(g);
    ns = 1;
    train_idx = Train_idx{ns};
    idxs = [1:1:num_count];
    idxs(train_idx) = 0;
    test_idx = find(idxs ~= 0)';
    
    % this train and test idx refers to the sequences
    data_angles_seq_train = data_angles_seq_nonempty(train_idx);
    data_angles_seq_test = data_angles_seq_nonempty(test_idx);
    
    % Train the CRBM
    % Arrange all sequences into one matrix and with Size as sequence number
    % and train CRBM (with joint angles/pose or concatenated version as input)
    a_autoencoder_inputs = [];
    Size = [];
    for ii=1:1:length(data_angles_seq_train)
        % get the training sequence
        train_seq_motion_vector = data_angles_seq_train{ii}{4};
        
        num_frames = size(train_seq_motion_vector,1);
        
        P_desc_b = train_seq_motion_vector(1:num_frames,:);
        
        a_autoencoder_inputs = [a_autoencoder_inputs ; P_desc_b];
        %Size = [Size ; num_frames];
        Size = [Size ; size(P_desc_b,1)];
    end
    
    % Train the autoencoder
    opts_auto.seq_size = Size;
    st = stacked_autoencoder(a_autoencoder_inputs,opts_auto,true);
    
    g=gpuDevice;
    reset(g);
    
    st_all{ns} = st;
    
    % setting the size
    opts_crbm.seq_lengths = Size;
    
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
    
    %get the crbm structure from crbm_gpu
    crbm.total_num_cases = crbm_gpu.total_num_cases;
    crbm.num_input_dims = crbm_gpu.num_input_dims;
    crbm.sizes = crbm_gpu.sizes;
    crbm.num_layers = crbm_gpu.num_layers;
    crbm.gsd = crbm_gpu.gsd;
    crbm.modes = crbm_gpu.modes;
    crbm.mode_str = crbm_gpu.mode_str;
    crbm.dropRate = crbm_gpu.dropRate;
    crbm.numGibbs = crbm_gpu.numGibbs;
    for ll = 1:1:length(crbm_gpu.rbm)
        crbm.rbm{ll}.nt = crbm_gpu.rbm{ll}.nt;
        crbm.rbm{ll}.num_dims = crbm_gpu.rbm{ll}.num_dims;
        crbm.rbm{ll}.num_hid = crbm_gpu.rbm{ll}.num_hid;
        crbm.rbm{ll}.w = gather(crbm_gpu.rbm{ll}.w);
        crbm.rbm{ll}.bi = gather(crbm_gpu.rbm{ll}.bi);
        crbm.rbm{ll}.bj = gather(crbm_gpu.rbm{ll}.bj);
        crbm.rbm{ll}.A = gather(crbm_gpu.rbm{ll}.A);
        crbm.rbm{ll}.B = gather(crbm_gpu.rbm{ll}.B);
        crbm.rbm{ll}.w_update = gather(crbm_gpu.rbm{ll}.w_update);
        crbm.rbm{ll}.bi_update = gather(crbm_gpu.rbm{ll}.bi_update);
        crbm.rbm{ll}.bj_update = gather(crbm_gpu.rbm{ll}.bj_update);
        crbm.rbm{ll}.A_update = gather(crbm_gpu.rbm{ll}.A_update);
        crbm.rbm{ll}.B_update = gather(crbm_gpu.rbm{ll}.B_update);
        crbm.rbm{ll}.epsilon_w = crbm_gpu.rbm{ll}.epsilon_w;
        crbm.rbm{ll}.epsilon_bi = crbm_gpu.rbm{ll}.epsilon_bi;
        crbm.rbm{ll}.epsilon_bj = crbm_gpu.rbm{ll}.epsilon_bj;
        crbm.rbm{ll}.epsilon_A = crbm_gpu.rbm{ll}.epsilon_A;
        crbm.rbm{ll}.epsilon_B = crbm_gpu.rbm{ll}.epsilon_B;
        crbm.rbm{ll}.w_decay = crbm_gpu.rbm{ll}.w_decay;
        crbm.rbm{ll}.momentum = crbm_gpu.rbm{ll}.momentum;
        crbm.rbm{ll}.num_epochs = crbm_gpu.rbm{ll}.num_epochs;
        crbm.rbm{ll}.gsd = crbm_gpu.rbm{ll}.gsd;
    end
    crbm.batchsize = crbm_gpu.batchsize;
    crbm.data_mean = gather(crbm_gpu.data_mean);
    crbm.data_std = gather(crbm_gpu.data_std);
    crbm.seq_lengths = crbm_gpu.seq_lengths;
    
    g = gpuDevice;
    reset(g);
    
    crbm_all{ns} = crbm;
    
%end

save -mat gaitAnalysis_Trial1_Workspace st crbm
