% Testing_DeepKinematic_Network_Ver1
P_desc_deepNetwork_Proposed_prob_Overall = []; 
P_desc_deepNetwork_Proposed_prob_Overall_true = []; 
class_labels_vest_novest_Overall = [];
train_idx = Train_idx{ns};
idxs = [1:1:num_count];
idxs(train_idx) = 0;
test_idx = find(idxs ~= 0)';
    
% this train and test idx refers to the sequences
data_angles_seq_train = data_angles_seq_nonempty(train_idx);
data_angles_seq_test = data_angles_seq_nonempty(test_idx);


% for the complete training set
for ii=1:1:length(data_angles_seq_train)
    % get the training sequence
    train_seq_jangles = data_angles_seq_train{ii}{1};
    train_seq_pos = data_angles_seq_train{ii}{2};
    train_class_labels = data_angles_seq_train{ii}{3};
    train_seq_motion_vector = data_angles_seq_train{ii}{4};
    train_seq_image_vector = data_angles_seq_train{ii}{5};
    
    num_frames = size(train_seq_motion_vector,1);
    
    P_desc_b = train_seq_image_vector(1:num_frames,:);
    num_dims = size(train_seq_image_vector,2);
    try
        motionVectorTrain = train_seq_motion_vector(1:num_frames,:);
    catch exception
        disp(exception)
    end
    imageVectorTrain = train_seq_image_vector(1:num_frames,:);
    kinematicModelVectorTrain = [train_seq_pos train_seq_jangles];
    
    % apply the normalization
    motionVectorTrain = (motionVectorTrain - min_motionDisp)./(max_motionDisp - min_motionDisp);
    imageVectorTrain = (imageVectorTrain - min_imageDisp)./(max_imageDisp - min_imageDisp);
    
    kinematicModelVectorTrainEst = deep_gait_regression_test(dp,motionVectorTrain,imageVectorTrain,true);
    P_desc_deepNetwork_Proposed_prob_Overall = [P_desc_deepNetwork_Proposed_prob_Overall ; kinematicModelVectorTrainEst]; 
    P_desc_deepNetwork_Proposed_prob_Overall_true = [P_desc_deepNetwork_Proposed_prob_Overall_true ; kinematicModelVectorTrain]; 
    class_labels_vest_novest_Overall = [class_labels_vest_novest_Overall; train_class_labels];
end

%% find the mean kinematic model trajector
idxs_1 = (class_labels_vest_novest_Overall == 1);
idxs_2 = (class_labels_vest_novest_Overall == 2);
joint_angle_traj_man_Proposed_prob_Class1 = P_desc_deepNetwork_Proposed_prob_Overall(idxs_1,5:6);
joint_angle_traj_man_Proposed_prob_Class1_true = P_desc_deepNetwork_Proposed_prob_Overall_true(idxs_1,5:6);

joint_angle_traj_man_Proposed_prob_Class2 = P_desc_deepNetwork_Proposed_prob_Overall(idxs_2,5:6);
joint_angle_traj_man_Proposed_prob_Class2_true = P_desc_deepNetwork_Proposed_prob_Overall_true(idxs_2,5:6);

% Display flags for estimated kinematic model vector

num_of_angles = size(joint_angle_traj_man_Proposed_prob_Class1,2);
num_of_frames = size(joint_angle_traj_man_Proposed_prob_Class1,1);
figure(1);
for jo = 1:1:num_of_angles
   plot(1:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob_Class1(:,jo)),'r','LineWidth',1.5);
   hold on;
end
hold on;
num_of_angles = size(joint_angle_traj_man_Proposed_prob_Class1_true,2);
num_of_frames = size(joint_angle_traj_man_Proposed_prob_Class1_true,1);
for jo = 1:1:num_of_angles
   plot(1:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob_Class1_true(:,jo)),'b','LineWidth',1.5);
   hold on;
end
xlabel('Frame Number','FontSize',10,'FontWeight','bold');
ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
s_title = sprintf('Joint Angle Trajectories true and estimated for Class 1');
title(s_title,'FontSize',10,'FontWeight','bold');
set(gca,'FontSize',10,'FontWeight','bold');
hold off

num_of_angles = size(joint_angle_traj_man_Proposed_prob_Class2,2);
num_of_frames = size(joint_angle_traj_man_Proposed_prob_Class2_true,1);
figure(2);
for jo = 1:1:num_of_angles
   plot(1:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob_Class2(:,jo)),'r','LineWidth',1.5);
   hold on;
end
hold on;
for jo = 1:1:num_of_angles
   plot(1:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob_Class2_true(:,jo)),'b','LineWidth',1.5);
   hold on;
end
xlabel('Frame Number','FontSize',10,'FontWeight','bold');
ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
s_title = sprintf('Joint Angle Trajectories true and estimated for Class 2');
title(s_title,'FontSize',10,'FontWeight','bold');
set(gca,'FontSize',10,'FontWeight','bold');
hold off

%%

train_vectors = double(P_desc_deepNetwork_Proposed_prob_Overall);
labels_train = double(class_labels_vest_novest_Overall);
P_desc_deepNetwork_Proposed_prob_Overall = []; 
class_labels_vest_novest_Overall = [];

%% CREATE THE GRAPH WHICH TELLS THE PREDICTION PERFORMANCE OF THE NETWORK

% for the test set
for ii=1:1:length(data_angles_seq_test)
    % get the training sequence
    test_seq_jangles = data_angles_seq_test{ii}{1};
    test_seq_pos = data_angles_seq_test{ii}{2};
    test_class_labels = data_angles_seq_test{ii}{3};
    test_seq_motion_vector = data_angles_seq_test{ii}{4};
    test_seq_image_vector = data_angles_seq_test{ii}{5};
    
    num_frames = size(test_seq_motion_vector,1);
    
    P_desc_b = test_seq_image_vector(1:num_frames,:);
    num_dims = size(test_seq_image_vector,2);
    try
        motionVectorTest = test_seq_motion_vector(1:num_frames,:);
    catch exception
        disp(exception)
    end
    imageVectorTest = test_seq_image_vector(1:num_frames,:);
    kinematicModelVectorTest = [test_seq_pos test_seq_jangles];
    
    % apply the normalization
    motionVectorTest = (motionVectorTest - min_motionDisp)./(max_motionDisp - min_motionDisp);
    imageVectorTest = (imageVectorTest - min_imageDisp)./(max_imageDisp - min_imageDisp);
    
    kinematicModelVectorTestEst = deep_gait_regression_test(dp,motionVectorTest,imageVectorTest,true);
    P_desc_deepNetwork_Proposed_prob_Overall = [P_desc_deepNetwork_Proposed_prob_Overall ; kinematicModelVectorTestEst]; 
    class_labels_vest_novest_Overall = [class_labels_vest_novest_Overall; test_class_labels];
end

test_vectors = double(P_desc_deepNetwork_Proposed_prob_Overall);
labels_test = double(class_labels_vest_novest_Overall);

% Do SVM Classification of the joint angles
% RUN the SVM on the appropriate training data/testing data
%% Applying the SVM Classifier
% setting the options
% Using precomputed kernel
C_val = 0.5; % Tried different values of C = 0.5, 1, 1.5
opts = sprintf('-s 0 -t 4 -c %d -b 1 -q',C_val);
%opts = sprintf('-s 0 -t 0 -c %d -b 1 -q',C_val);
total_N = size(train_vectors,1);

% precomputing the kernel
K_train = slmetric_pw(train_vectors',train_vectors','chisq');
A_mean = mean(K_train(:));      

K_combined = exp(-1 * (1/A_mean * K_train));
% Multi-Channel Kernel
%         K_combined = exp(-1 * ( 1/A_mean1 * K_train1 + 1/A_mean2 * K_train2 + 1/A_mean3 * K_train3));

data = [ (1:size(train_vectors,1))' , K_combined];

K_test = slmetric_pw(test_vectors',train_vectors','chisq');
% training the svm
model = libsvmtrain(labels_train,data,opts);

% computing the test kernel matrix

K_combined_test = exp(-1 * (1/A_mean * K_test));
%         K_combined_test = exp(-1 * (1/A_mean1 * K_test1 + 1/A_mean2 * K_test2 + 1/A_mean3 * K_test3));

data_test = [ (1:size(test_vectors,1))' , K_combined_test];
%data_test = test_vectors;

% predicting the labels
[predicted_label,accuracy,dec_vals] = libsvmpredict(labels_test,data_test,model);
fi = 1;
acc_per_sim(fi,ns) = accuracy(1);
%% Nearest neighbhor
% Test using nearest neighbor classifier
num_neighbors = 1;
[ids_org,dis] = yael_nn(single(train_vectors'),single(test_vectors'),num_neighbors,3);

if(num_neighbors > 1)
    predicted_label2_org = (sum(labels_train(ids_org))/num_neighbors) > 0.5;
    predicted_label2_org = predicted_label2_org';
else
    predicted_label2_org = labels_train(ids_org); 
end

accuracy2_org = 1 - sum(xor(labels_test,predicted_label2_org))/length(labels_test);
