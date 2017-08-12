%% Script to train a deep neural network for function approximation
% set the parameters
N_sim = 1;
opts_auto.epsilon_w = 0.1;
opts_auto.epsilon_vb = 0.1;
opts_auto.epsilon_vc = 0.1;
opts_auto.momentum = 0.9;
opts_auto.weightcost = 0.0001;
opts_auto.numepochs = 1000;
opts_auto.sizes = [100 40];
opts_auto.batch_size = 100;

% load the data_angles

[num_of_subjects,num_of_phases] = size(data_angles);
% extract the elements of the data_angles and organize them into
% joint_angles, pos, and class 

joint_angles = [];
pos_vector = [];
class_vector = [];
for sbj=1:1:num_of_subjects
    for phj = 1:1:num_of_phases % trying out only one phase
        if(isempty(data_angles{sbj,phj}))
            continue;
        end
        joint_angles = [joint_angles ; data_angles{sbj,phj}{1}];
        pos_vector = [pos_vector; data_angles{sbj,phj}{2}];
        class_vector = [class_vector; data_angles{sbj,phj}{3}];
    end
end

total_N = size(class_vector,1);

% organize the data into training and testing phase
for ns = 1:1:N_sim
    class1_idxs = find(class_vector == 1);
    class2_idxs = find(class_vector == 2);
    
    % SPLIT THE TRAINING/TESTING FROM EACH CLASS : NO UNEVEN DISTRIBUTION OF TRAIN VECTORS
    train_idx1 = randsample(class1_idxs,ceil(2/3 * length(class1_idxs)));
    train_idx2 = randsample(class2_idxs,ceil(2/3 * length(class2_idxs)));
    train_idx = [train_idx1 ; train_idx2];
    idxs = [1:1:total_N];
    idxs(train_idx) = 0;
    test_idx = find(idxs ~= 0)';
    
    labels_train = class_vector(train_idx);
    labels_test = class_vector(test_idx);
    
    % get the training samples
    joint_angles_train = joint_angles(train_idx,:);
    pos_vector_train = pos_vector(train_idx,:);
    
    % normalize the data between the range of 0 and 1
    norm_factor_pos_min = min(pos_vector_train(:));
    norm_factor_pos_max = max(pos_vector_train(:));
    
    pos_vector_train = (pos_vector_train - norm_factor_pos_min)./(norm_factor_pos_max - norm_factor_pos_min);  
    joint_angles_train = (joint_angles_train - (-pi))./(2*pi);
    
    % train the network
    fn = nn_train(pos_vector_train,joint_angles_train,opts_auto,false);
    
    
end


% set the options of the deep neural network

% train the function approximator with a single hidden layer deep network
