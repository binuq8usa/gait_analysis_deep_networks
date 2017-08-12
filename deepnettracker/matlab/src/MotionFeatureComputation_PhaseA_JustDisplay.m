%% Script file to load the tracked joint locations of all sequences belonging to all the phases
% Setting the path variables 
clear all;

% HOME_PATH = getenv('HOME'); 
% DATASET_PATH = sprintf('%s/Datasets/GaitAnalysis/AFITUnclassifiedInformation',HOME_PATH);
% OPTICAL_FLOW_PATH = sprintf('%s/Datasets/GaitAnalysis/AFITUnclassifiedInformation/OpticalFlow',HOME_PATH);
% 
% MANUAL_JOINT_TRACKS_PATH = sprintf('%s/Datasets/GaitAnalysis/AFITUnclassifiedInformation',HOME_PATH);

HOME_PATH = getenv('HOME'); 
GAITANALYSIS_PATH = '/Datasets/GaitAnalysis';
DATASET_PATH = sprintf('%s%s/AFITUnclassifiedInformation',HOME_PATH,GAITANALYSIS_PATH);
OPTICAL_FLOW_PATH = sprintf('%s%s/AFITUnclassifiedInformation/OpticalFlow',HOME_PATH,GAITANALYSIS_PATH);
TRACKED_JOINT_TRACKS_PATH = strcat(DATASET_PATH,'/JointTracks');
MANUAL_JOINT_TRACKS_PATH = DATASET_PATH;

% Getting the list of subjects
list_of_subjects = dir(fullfile(DATASET_PATH,'/Subject*'));
Start_frames = [];
Filenames = [];
Detections = {};

% Iterating through each subject
for k = 1:1:length(list_of_subjects)
    subject_name = list_of_subjects(k).name;
    subject_path = sprintf('%s/Deinterlaced/%s',DATASET_PATH,subject_name);
    subject_optflow_path = sprintf('%s/%s',OPTICAL_FLOW_PATH,subject_name);
    
    subject_tracked_joint_path = strcat(TRACKED_JOINT_TRACKS_PATH,'/',subject_name);
    
%     subject_fg_path = sprintf('%s/%s',FOREGROUND_PATH,subject_name);
    subject_manual_joint_path = strcat(MANUAL_JOINT_TRACKS_PATH,'/',subject_name);
    
    list_of_phases = dir(fullfile(subject_path,'/PHASE_*'));
    
    % Iterating through each phase
    for l = 1:1:length(list_of_phases)
        phase_name = list_of_phases(l).name;
        phase_name
        phase_video_path = sprintf('%s/%s',subject_path,phase_name);
        phase_optflow_path = sprintf('%s/%s',subject_optflow_path,phase_name);
        
        %        phase_fg_video_path = sprintf('%s/%s',subject_fg_path,phase_name);'
        phase_manual_joint_path = strcat(subject_manual_joint_path,'/',phase_name);
        phase_tracked_joint_path = strcat(subject_tracked_joint_path,'/',phase_name);
      
       list_of_video_files = dir(fullfile(phase_video_path,'/*.avi'));
       list_of_files_manual = dir(fullfile(phase_manual_joint_path,'/*.txt'));
       
       % Iterating through each file
       % do only the second video file
       for m = 1:1:length(list_of_video_files) % for each version number of files are the same
           
           video_file = sprintf('%s/%s',phase_video_path,list_of_video_files(m).name);
           name_video = list_of_video_files(m).name(1:end-10)
           obj_video = VideoReader(video_file);
           frames = read(obj_video,[1 inf]);
%            
% %            fg_video_file = sprintf('%s/%s',phase_fg_video_path,list_of_video_files(m).name);
% %            obj_video_fg = VideoReader(fg_video_file);
% %            frames_fg = read(obj_video_fg);
%            
%            num_frames = size(frames,4);
%            num_rows = size(frames,1);
%            num_cols = size(frames,2);
%            num_bands = size(frames,3);
           
           file_name_manual = list_of_files_manual(m).name;
           manual_pts_filename = strcat(phase_manual_joint_path,'/',file_name_manual);
           
           % Load the manual points 
           try
               fid_1 = fopen(manual_pts_filename);
               A = fscanf(fid_1,'%d',3);
               num_of_frames = A(1);
               num_of_joints = A(2);
               start_frame = A(3);
               A = fscanf(fid_1,'%f',[2*num_of_joints, num_of_frames]);
               ManualPts = A';
               fclose(fid_1);
           catch exception
               disp(exception)
               continue;
           end
           
           dpm_detector_file = sprintf('%s/Detections_%s',phase_video_path,file_name_manual);
           motion_descriptor_file = sprintf('%s/motionDescriptor_%s',phase_video_path,file_name_manual);
          
           % opening the dpm detector file
           fid = fopen(dpm_detector_file,'r');
           try
                A = fscanf(fid,'%g',[5 inf]);
           catch exception
               disp(exception);
               continue;
           end
           A = A'; % so that rows correspond to different frames
           % resize to a larger window
           A(:,2) = A(:,2) - A(:,4) * 20/100;
           A(:,3) = A(:,3) - A(:,5) * 12.5/100;
           A(:,4) = A(:,4) + A(:,4) * 40/100;
           A(:,5) = A(:,5) + A(:,5) * 25/100;
           fclose(fid);
                    
           %fid_m = fopen(motion_descriptor_file,'w');
           
              
           start_frame = A(1,1);
           num_of_frames = size(ManualPts,1);

           for fr_num = start_frame+1:1:start_frame + num_of_frames-1                
               
               % load the image
               img = frames(:,:,:,fr_num);
               
               % load the optical flow
               optMag_file = sprintf('%s/%s_optFlow_mag_framecount_%d.txt',phase_optflow_path,name_video,fr_num);
               optDir_file = sprintf('%s/%s_optFlow_dir_framecount_%d.txt',phase_optflow_path,name_video,fr_num);
               try
                   fileID = fopen(optMag_file);
                   str1 = fgetl(fileID);
                   str2 = fgetl(fileID);
                   size_img = str2num(str2);
                   img_mag = fscanf(fileID,'%f',[size_img(2) size_img(1)]);
                   img_mag = img_mag';
                   fclose(fileID);
                   
                   fileID = fopen(optDir_file);
                   str1 = fgetl(fileID);
                   str2 = fgetl(fileID);
                   size_img = str2num(str2);
                   img_dir = fscanf(fileID,'%f',[size_img(2) size_img(1)]);
                   img_dir = img_dir';
                   fclose(fileID);
               catch exception
                   disp(exception)
                   break % end of sequence : no more optical flow mag frames
               end

               % defining a crop size based on the discrete pose provided
               % in the dataset
               img_mask = zeros(size(img_mag,1),size(img_mag,2));
               try
                   x_min = min(ManualPts(fr_num - start_frame + 1,1:2:12))-12;
                   x_max = max(ManualPts(fr_num - start_frame + 1,1:2:12))+12;
                   y_min = min(ManualPts(fr_num - start_frame + 1,2:2:12))-12;
                   y_max = max(ManualPts(fr_num - start_frame + 1,2:2:12))+12;
               catch exception
                   disp(exception)
               end
               
               %imshow(img);
               %rectangle('Position',[x_min y_min x_max-x_min y_max-y_min]);
               
               try
                    %crop_size = [A(fr_num - start_frame + 1,2:5)];
                    crop_size = [x_min y_min x_max-x_min y_max-y_min];
               catch exception
                   disp(exception)
               end
               
%                img_dir = img_dir + 180;% range from 0 to 360
%                img_dir = img_dir/360; % value between 0 and 1
%                img_mag = img_mag/max(img_mag(:));
%                
%                % crop the image and the optical flow
% %                im = imcrop(img,crop_size);
                img_u = img_mag .* cosd(img_dir);
                img_v = img_mag .* sind(img_dir);
                
                %max_d = norm([max(abs(img_u(:))) max(abs(img_v(:)))]);
                %img_u = img_u / max_d;
                %img_v = img_v / max_d;

                im_u = imcrop(img_u,crop_size);
                im_v = imcrop(img_v,crop_size);
                im_mask = ones(size(im_u,1),size(im_u,2));
                im_mag = imcrop(img_mag,crop_size);
                im_dir = imcrop(img_dir,crop_size);
                
%                
%                %imshow(im_mag);
                im2_u = imresize(im_u,[64 32],'nearest');
                im2_v = imresize(im_v,[64 32],'nearest');
                im2_mask = imresize(im_mask,[64 32],'nearest');
                im2_mag = imresize(im_mag,[64 32],'nearest');
                im2_dir = imresize(im_dir,[64 32],'nearest');
%                figure(1);imshow(img_mag);
%                figure(2);imshow(im_mag);

                %if(max(abs(im2_u(:))) > 1 || max(abs(im2_v(:)) > 1) )
                %    disp('jssue');
                %end

               %im2_mag = (im2_mag - min(im2_mag(:))) / (max(im2_mag(:)) - min(im2_mag(:)));
               %im2_dir = (im2_dir - min(im2_dir(:))) / (max(im2_dir(:)) - min(im2_dir(:)));
               
               
               % not enough samples to train the autoencoder : curse of
               % dimensionality
               %H = [abs(im2_u(:)); abs(im2_v(:))]; % use absolute value since direction of movement is not considered and to make it between 0 and 1
               H = [im2_u(:); im2_v(:)];
               
               % Compute HHOF vector
               % Get the Radon transform vector and the LBFP vector along
               % with HOF. Use that information in the EI journal for
               % motion computation.
               % Since there are only a few samples, to avoid
               % dimensionality reduction, we employ hand-engineered
               % features.
               % We will use autoencoder to learn higher level
               % representations of those hand-engineered features.
               % Directly learning from optical flow requires a lot of
               % features
               %[H,L] = ComputeLBPHOF(im_mag,im_dir,im_mask,true); % cropped version of the mask
               %H = H';
               % resize to 64 x 32 size and concatentate them into two
               % vectors
               
               
%                figure(3); plot(H);
%                % save HHOF vector
%                try
%                     fprintf(fid_m,'%f \t',H);
%                     fprintf(fid_m,'\n');
%                catch exception
%                    disp(exception)
%                end
            end
            %fclose(fid_m);
           
           Filenames = [Filenames ; {file_name_manual(1:length(file_name_manual)-10)}];
           Start_frames = [Start_frames ; start_frame];
           
           s_mesg = sprintf('%s processed',motion_descriptor_file);
           disp(s_mesg);
           
       end
       
       
    end
    
end
