% sample autoencoder and CRBM testing
% load -mat Workspace_Autoencoder_CRBM_Structure_Trail1
% st_x = st_all{1,1};
% st_y = st_all{1,2};
% crbm = crbm_all{1};


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
    num_dims = size(train_seq_motion_vector,2);
        
    motionVector_ux = motionVector(:,1:num_dims/2);
    motionVector_uy = motionVector(:,num_dims/2+1:end);
        
    [X_out, motionLatentVector_ux] = sae_nn_ff(st_x,motionVector_ux);
    [X_out, motionLatentVector_uy] = sae_nn_ff(st_y,motionVector_uy);
        
%         % tensor product of pose vector and joint angles
%         P_desc = [];
%         for jj = 1:1:num_frames
%             P_desc = [P_desc ; kron(pose_vector(jj,:),motionLatentVector(jj,:))];
%         end

    % set the state of the CRBM tracker
    P_desc = [];
    for jj = 1:1:num_frames
        P_desc = [P_desc ; [pose_vector(jj,:) motionLatentVector_ux(jj,:) motionLatentVector_uy(jj,:)]]; % tensor sum or concatenation
    end
    
    a_crbm_train_inputs = P_desc;
    
    num_frames = size(a_crbm_train_inputs,1);
    
    if(num_frames <= 2*opts_crbm.nt)
        continue; % do not include in the vector
    end
    
    labels_train = [labels_train; class_vector(1)*ones(size(a_crbm_train_inputs,1),1)];
    
    % Normalize the test features obtaining CRBM representation
    a_crbm_train_inputs_norm = (a_crbm_train_inputs - repmat(crbm.data_mean,num_frames,1))./(repmat(crbm.data_std,num_frames,1));
    
    n_t = 10;
    % Applying the crbm
    for start_fr_num = 1:1:num_frames-n_t+1
        [a_crbm_train_inputs_norm_rec,train_l1_rep,train_l1_rep] = testCRBM(crbm,3*n_t,a_crbm_train_inputs_norm,start_fr_num);
        
%         % get the unnormalized version from the regenerated sequence
%         a_X_test_out_rec = (a_X_test_out_rec_norm .* repmat(crbm_local.data_std,L_dash,1)) + repmat(crbm_local.data_mean,L_dash,1);
%         
%         % apply the decoder
%         X_test_out_rec = sae_ff_nn_decoder(st,a_X_test_out_rec);
%         X1 = X_test(start_fr_num+2*nt:1:L_dash + start_fr_num-1,:);
%         X2 = X_test_out_rec(2*nt+1:L_dash,:);
    end
    
    % apply the trained CRBM
    [a_crbm_train_inputs_norm_rec,train_l1_rep,train_l2_rep] = testCRBM(crbm,num_frames,a_crbm_train_inputs_norm,1);
    
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
    num_dims = size(train_seq_motion_vector,2);
        
    motionVector_ux = motionVector(:,1:num_dims/2);
    motionVector_uy = motionVector(:,num_dims/2+1:end);
        
    [X_out, motionLatentVector_ux] = sae_nn_ff(st_x,motionVector_ux);
    [X_out, motionLatentVector_uy] = sae_nn_ff(st_y,motionVector_uy);
        
%         % tensor product of pose vector and joint angles
%         P_desc = [];
%         for jj = 1:1:num_frames
%             P_desc = [P_desc ; kron(pose_vector(jj,:),motionLatentVector(jj,:))];
%         end

        % set the state of the CRBM tracker
    P_desc = [];
    for jj = 1:1:num_frames
        P_desc = [P_desc ; [pose_vector(jj,:) motionLatentVector_ux(jj,:) motionLatentVector_uy(jj,:)]]; % tensor sum or concatenation
    end
    
    %a_crbm_test_inputs = ComputeShapeOfTrajectory(test_seq_jangles, win_size);
    %a_crbm_test_inputs = test_seq_motion_vector;
    a_crbm_test_inputs = P_desc;

    num_frames = size(a_crbm_test_inputs,1);
    
    if(num_frames <= 2*opts_crbm.nt)
        continue; % do not include in the vector
    end
    
    labels_test = [labels_test; class_vector(1)*ones(size(a_crbm_test_inputs,1),1)];
    
    %% TODO: Its only checking the first 10 frames, and we are comparing the generated features as opposed to actual ones at that frame. 
    %% so, we should repeat for every frame
    %% right now, it takes only the first 10 + 1 frames of a sequence and the rest of the frames are generated.  
    %% we need to generate only one frame and keep shifting it like we did for dissertation.
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
num_neighbors = 10;
[ids_org,dis] = yael_nn(single(train_vectors_org'),single(test_vectors_org'),num_neighbors,3);

if(num_neighbors > 1)
    predicted_label2_org = (sum(labels_train(ids_org))/num_neighbors) > 0.5;
    predicted_label2_org = predicted_label2_org';
else
    predicted_label2_org = labels_train(ids_org); 
end

accuracy2_org = 1 - sum(xor(labels_test,predicted_label2_org))/length(labels_test);

[ids_l1,dis] = yael_nn(single(train_vectors_l1'),single(test_vectors_l1'),num_neighbors,3);

if(num_neighbors > 1)
    predicted_label2_l1 = (sum(labels_train(ids_l1))/num_neighbors) > 0.5;
    predicted_label2_l1 = predicted_label2_l1';
else
    predicted_label2_l1 = labels_train(ids_l1);
end

accuracy2_l1 = 1 - sum(xor(labels_test,predicted_label2_l1))/length(labels_test);

[ids_l2,dis] = yael_nn(single(train_vectors_l2'),single(test_vectors_l2'),num_neighbors,3);

if(num_neighbors > 1)
    predicted_label2_l2 = (sum(labels_train(ids_l2))/num_neighbors) > 0.5;
    predicted_label2_l2 = predicted_label2_l2';
else
    predicted_label2_l2 = labels_train(ids_l2); 
end

accuracy2_l2 = 1 - sum(xor(labels_test,predicted_label2_l2))/length(labels_test);

acc(ns) = accuracy(1);
acc2_org(ns) = accuracy2_org;
acc2_l1(ns) = accuracy2_l1;
acc2_l2(ns) = accuracy2_l2;

%% Nearest neigbhor
num_neighbors = 1;
[ids_org,dis] = yael_nn(single(train_vectors'),single(test_vectors'),num_neighbors,3);

if(num_neighbors > 1)
    predicted_label2_org = (sum(labels_train(ids_org))/num_neighbors) > 0.5;
    predicted_label2_org = predicted_label2_org';
else
    predicted_label2_org = labels_train(ids_org); 
end

accuracy2_org = 1 - sum(xor(labels_test,predicted_label2_org))/length(labels_test);
