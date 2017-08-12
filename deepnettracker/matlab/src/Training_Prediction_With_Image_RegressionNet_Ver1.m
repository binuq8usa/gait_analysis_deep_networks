%% Script to train a deep neural network for function approximation
% set the parameters
% TODO: Try using rectilinear units in autoencoder training and backpropagation
%% Tried the deep regression neural network with and without rectilinear units.
% But, the algorithm will only diverge.
% the only way is to augment the input vector x with the output target
% vector y to learn the inverse kinematic relationship.
% The hidden representations will give us the kinematic latent vector 
% This can be combined with the motion latent vector
% Both can be combined in a CRBM 
% Then, we can error-backpropagation where the objective function is a
% combination to classify the latent gait features as a threat or
% non-threat.

%% 
N_sim = 1;

opts_auto.epsilon_w = 0.1;
opts_auto.epsilon_vb = 0.1;
opts_auto.epsilon_vc = 0.1;
opts_auto.momentum = 0.9;
opts_auto.weightcost = 0.0001;
opts_auto.numepochs = 100;
opts_auto.sizes = [2048 512 64]; %  layers divided by 4
opts_auto.batch_size = 200;

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

%%
Train_idx = cell(N_sim,1);
for ns = 1:1:N_sim
    class1_idxs = find(class_labels_seq == 1);
    class2_idxs = find(class_labels_seq == 2);
    
    % SPLIT THE TRAINING/TESTING FROM EACH CLASS : NO UNEVEN DISTRIBUTION OF TRAIN VECTORS
    train_idx1 = randsample(class1_idxs,ceil(9/10 * length(class1_idxs)));
    train_idx2 = randsample(class2_idxs,ceil(9/10 * length(class2_idxs)));
    train_idx = [train_idx1 ; train_idx2];
    Train_idx{ns} = train_idx;
end
%%
st_image = cell(N_sim,1);

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
    a_autoencoder_inputs_im = [];
    
    Size = [];
    target_vector = [];
    for ii=1:1:length(data_angles_seq_train)
        % get the training sequence
        train_seq_jangles = data_angles_seq_train{ii}{1};
        train_seq_pos = data_angles_seq_train{ii}{2};
        train_seq_image_vector = data_angles_seq_train{ii}{5};
        
        num_frames = size(train_seq_image_vector,1);
        
        P_desc_b = train_seq_image_vector(1:num_frames,:);
        num_dims = size(train_seq_image_vector,2);
        
        a_autoencoder_inputs_im = [a_autoencoder_inputs_im ; P_desc_b];
        
        target_vector = [target_vector ; train_seq_pos train_seq_jangles];
        
        %Size = [Size ; num_frames];
        Size = [Size ; size(P_desc_b,1)];
    end
    
    % Train the autoencoder
    opts_auto.seq_size = Size;
    
    fn = rectifier_nn_train(a_autoencoder_inputs_im,target_vector,opts_auto,true);
    
    %st_im = stacked_autoencoder(a_autoencoder_inputs_im,opts_auto,true);
    
 