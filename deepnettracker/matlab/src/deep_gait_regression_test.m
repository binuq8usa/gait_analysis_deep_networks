%% Function Description
% This function trains a deep gait network which incorporates motion
% components, pose components, kinematic model-based joint angle
% components, image components to classify a gait as a threat or non-threat
function kinematicModelVector = deep_gait_regression_test(dp,motionVectorTest, imageVectorTest,flag_whiten)

%% Initialization PHASE
% Initialize the random number generator
rand('state',0);
num_samples = size(motionVectorTest,1);
numDimMotion = size(motionVectorTest,2);
numDimImage = size(imageVectorTest,2);

% Whitening the data 
if(flag_whiten)
    [motionVectorTestWhite,motionVectorTest] = pcaWhitenWithoutSVD(motionVectorTest,dp.numRedDimMotion, dp.U_motion, dp.S_motion,dp.motionVectorMean);
    [imageVectorTestWhite,imageVectorTest] = pcaWhitenWithoutSVD(imageVectorTest,dp.numRedDimImage, dp.U_image, dp.S_image,dp.imageVectorMean);
    motionVectorTestNorm = (motionVectorTest - dp.motionVectorMin)./(dp.motionVectorMax - dp.motionVectorMin);
    imageVectorTestNorm = (imageVectorTest - dp.imageVectorMin)./(dp.imageVectorMax - dp.imageVectorMin);
else
%     motionVectorTest = imresize(motionVectorTest,0.25,'nearest');
%     imageVectorTest = imresize(imageVectorTest,0.25,'nearest');
    motionVectorTestNorm = motionVectorTest;
    imageVectorTestNorm = imageVectorTest;
end

% applying the DBN network layer
[motionLatentVectorTest] = dbn_forward(motionVectorTestNorm, dp.R_motion_dbn);
[imageLatentVectorTest] = dbn_forward(imageVectorTestNorm, dp.R_image_dbn);

g=gpuDevice;
reset(g);

% apply the kinematic model neural network
inputLatentVectorTest = [motionLatentVectorTest imageLatentVectorTest];
v0Test = gpuArray(single(inputLatentVectorTest));

[cl,kinematicModelVector] = mlp_classify(dp.mlp_regression_network, v0Test);

kinematicModelVector = gather(kinematicModelVector);

reset(g)

end
