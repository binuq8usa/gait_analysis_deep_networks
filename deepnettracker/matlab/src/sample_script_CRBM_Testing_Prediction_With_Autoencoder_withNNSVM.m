% sample autoencoder and CRBM testing
% get the crbm structure from crbm_gpu
% crbm.total_num_cases = crbm_gpu.total_num_cases;
% crbm.num_input_dims = crbm_gpu.num_input_dims;
% crbm.sizes = crbm_gpu.sizes;
% crbm.num_layers = crbm_gpu.num_layers;
% crbm.gsd = crbm_gpu.gsd;
% crbm.modes = crbm_gpu.modes;
% crbm.mode_str = crbm_gpu.mode_str;
% crbm.dropRate = crbm_gpu.dropRate;
% crbm.numGibbs = crbm_gpu.numGibbs;
% for ll = 1:1:length(crbm_gpu.rbm)
%     crbm.rbm{ll} = gather(crbm_gpu.rbm{ll});
% end
% crbm.batchsize = crbm_gpu.batchsize;
% crbm.data_mean = gather(crbm_gpu.data_mean);
% crbm.data_std = gather(crbm_gpu.data_std);
% crbm.seq_lengths = crbm_gpu.seq_lengths;

% Getting the CRBM representations of the training sequences
svm_train_inputs = [];
labels_train = [];
train_vectors_l1 = [];
train_vectors_l2 = [];
train_vectors_org = [];
for ii =1:1:length(data_angles_seq_train)
    
    % get the features from the test sequence
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
    
    
    a_crbm_train_inputs = P_desc;
    
    num_frames = size(a_crbm_train_inputs,1);
    
    if(num_frames <= 2*opts_crbm.nt)
        continue; % do not include in the vector
    end
    
    labels_train = [labels_train; class_vector(1)*ones(size(a_crbm_train_inputs,1),1)];
    
    % Normalize the test features obtaining CRBM representation
    a_crbm_train_inputs_norm = (a_crbm_train_inputs - repmat(crbm.data_mean,num_frames,1))./(repmat(crbm.data_std,num_frames,1));
    
    % apply the trained CRBM
    [a_crbm_train_inputs_norm,train_l1_rep,train_l2_rep] = testCRBM(crbm,num_frames,a_crbm_train_inputs_norm,1);
    
    % accumulating the SVM train inputs : can change this to test the
    % effectiveness of the CRBM ( CHECK THE TESTING SIDE AS WELL)
    svm_train_inputs = [svm_train_inputs ; train_l2_rep];
    
    train_vectors_org = [train_vectors_org ; a_crbm_train_inputs];
    train_vectors_l1 = [train_vectors_l1 ; train_l1_rep];
    train_vectors_l2 = [train_vectors_l2; train_l2_rep];
    
end

%  TRAIN THE SVM CLASSIFIER
train_vectors = svm_train_inputs;

labels_train = labels_train - 1; % making it labels 0 and 1


K_train = slmetric_pw(train_vectors',train_vectors','chisq');
A_mean = mean(K_train(:));

K_combined = exp(-1 * (1/A_mean * K_train));

data = [ (1:size(train_vectors,1))' , K_combined];

model = libsvmtrain(labels_train,data,opts);

% Train the Nearest Neighbor classifier

% TEST the CRBM
labels_test = [];
svm_test_inputs = [];
test_vectors_org = [];
test_vectors_l1 = [];
test_vectors_l2 = [];
for ii =1:1:length(data_angles_seq_test)
    class_vector = data_angles_seq_test{ii}{3};
    %labels_test = [labels_test; class_vector];
    
    % get the features from the test sequence
    test_seq_jangles = data_angles_seq_test{ii}{1};
    test_seq_pos = data_angles_seq_test{ii}{2};
    %a_crbm_test_inputs = [test_seq_pos test_seq_jangles];
    
    test_seq_motion_vector = data_angles_seq_test{ii}{4};
    
    num_frames = size(test_seq_pos,1);
    
    pose_vector = [test_seq_pos test_seq_jangles];
    
    % accumulate the input data
    %a_crbm_train_inputs = [a_crbm_train_inputs ; [train_seq_pos train_seq_jangles]];
    
    % TODO: Need to combine motion vector with joint trajectory
    % Not the shape descriptro
    motionVector = test_seq_motion_vector(1:num_frames,:);
    
    [X_out, motionLatentVector] = sae_nn_ff(st,motionVector);
    
    % tensor product of pose vector and joint angles
    P_desc = [];
    for jj = 1:1:num_frames
        P_desc = [P_desc ; kron(pose_vector(jj,:),motionLatentVector(jj,:))];
    end
    
    a_crbm_test_inputs = P_desc;
    
    %a_crbm_test_inputs = ComputeShapeOfTrajectory(test_seq_jangles, win_size);
    %a_crbm_test_inputs = test_seq_motion_vector;
    
    num_frames = size(a_crbm_test_inputs,1);
    
    if(num_frames <= 2*opts_crbm.nt)
        continue; % do not include in the vector
    end
    
    labels_test = [labels_test; class_vector(1)*ones(size(a_crbm_test_inputs,1),1)];
    
    % Normalize the test features obtaining CRBM represent tion
    a_crbm_test_inputs_norm = (a_crbm_test_inputs - repmat(crbm.data_mean,num_frames,1))./(repmat(crbm.data_std,num_frames,1));
    
    % apply the trained CRBM
    [a_crbm_test_inputs_norm,test_l1_rep,test_l2_rep] = testCRBM(crbm,num_frames,a_crbm_test_inputs_norm,1);
    
    % accumulating the svm test inputs
    svm_test_inputs = [svm_test_inputs ; test_l2_rep];
    
    test_vectors_org = [test_vectors_org ; a_crbm_test_inputs];
    test_vectors_l1 = [test_vectors_l1 ; test_l1_rep];
    test_vectors_l2 = [test_vectors_l2 ; test_l2_rep];
end

% Test the SVM Classifier
test_vectors = svm_test_inputs;
labels_test = labels_test - 1; % making the labels 0 and 1

K_test = slmetric_pw(test_vectors',train_vectors','chisq');

K_combined_test = exp(-1 * (1/A_mean * K_test));
data_test = [ (1:size(test_vectors,1))' , K_combined_test];

% classify
[predicted_label,accuracy,dec_vals] = libsvmpredict(labels_test,data_test,model,'-b 1 -q');


%% STATE
% With original motion feature, accuracy is around 25 %
% with 50 feature units in layer 1: its around 49%
% with 50 feature units in layer 2: its around 51%
% Try with 150/200 units, increased number of iterations
% With 150/200 units, the accuracy is little better.
% KNN with neighbors 15, we get 58.8% for motion feature, 63.28 for l1
% and 62.30% for l2
% Do with Kron delta function and see the improvement.
% with learning rate increased to 10^-4, the accuracy drops to
% 33.76,40.27,47.74 with 2000 iterations,
% same accuracy as above with learning rate increased to 10^-3
% with 3000 iterations, new motion feature set, with learning rate set
% to 10^-5, the accuracies are 33.41%, 36.14%, 51.99%

%%
% Test using nearest neighbor classifier
num_neighbors = 15;
[ids_org,dis] = yael_nn(single(train_vectors_org'),single(test_vectors_org'),num_neighbors,3);

predicted_label2_org = (sum(labels_train(ids_org))/num_neighbors) > 0.5;

accuracy2_org = 1 - sum(xor(labels_test,predicted_label2_org'))/length(labels_test);

[ids_l1,dis] = yael_nn(single(train_vectors_l1'),single(test_vectors_l1'),num_neighbors,3);

predicted_label2_l1 = (sum(labels_train(ids_l1))/num_neighbors) > 0.5;

accuracy2_l1 = 1 - sum(xor(labels_test,predicted_label2_l1'))/length(labels_test);

[ids_l2,dis] = yael_nn(single(train_vectors_l2'),single(test_vectors_l2'),num_neighbors,3);

predicted_label2_l2 = (sum(labels_train(ids_l2))/num_neighbors) > 0.5;

accuracy2_l2 = 1 - sum(xor(labels_test,predicted_label2_l2'))/length(labels_test);

acc(ns) = accuracy(1);
acc2_org(ns) = accuracy2_org;
acc2_l1(ns) = accuracy2_l1;
acc2_l2(ns) = accuracy2_l2;