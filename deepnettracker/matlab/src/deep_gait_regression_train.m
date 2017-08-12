% function which will sample the deep belief nets for the motion and image
% vector and train an MLP(or something else) 
function dp = deep_gait_regression_train(dp,motionVector, kinematicModelVector, imageVector)

R_motion_dbn = dp.R_motion_dbn;
R_kinematic_dbn = dp.R_kinematic_dbn;

num_samples_motion = size(motionVector,1);
num_samples_image = size(imageVector,1);

motionLatentVector = dbn_sample(motionVector,R_motion_dbn,num_samples_motion);
imageLatentVector = dbn_sample(imageVector,R_kinematic_dbn,num_samples_image);

end