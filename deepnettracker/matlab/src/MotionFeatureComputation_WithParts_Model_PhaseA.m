%% Script file to load the tracked joint locations of all sequences belonging to all the phases
% Setting the path variables 
clear all;

%compile;

% load and display model
%load('PARSE_final');
% visualizemodel(model);
% disp('model template visualization');
% disp('press any key to continue'); 
% pause;
% visualizeskeleton(model);
% disp('model tree visualization');
% disp('press any key to continue'); 
% pause;
HOME_PATH = getenv('HOME'); 
DATASET_PATH = sprintf('%s/Datasets/AFITUnclassifiedInformation/Deinterlaced',HOME_PATH);
OPTICAL_FLOW_PATH = sprintf('%s/Datasets/AFITUnclassifiedInformation/OpticalFlow',HOME_PATH);

% FOREGROUND_PATH = '~/Datasets/AFITUnclassifiedInformation/Foreground';
MANUAL_JOINT_TRACKS_PATH = sprintf('%s/Datasets/AFITUnclassifiedInformation',HOME_PATH);
% Joint_names = {'Shoulder';'Elbow';'Wrist';'Hip';'Knee';'Ankle'};

% Getting the list of subjects
list_of_subjects = dir(fullfile(DATASET_PATH,'/Subject*'));
Start_frames = [];
Filenames = [];
Detections = {};

% Iterating through each subject
for k = 1:1:length(list_of_subjects)
    subject_name = list_of_subjects(k).name;
    subject_path = sprintf('%s/%s',DATASET_PATH,subject_name);
    subject_optflow_path = sprintf('%s/%s',OPTICAL_FLOW_PATH,subject_name);
    
%     subject_fg_path = sprintf('%s/%s',FOREGROUND_PATH,subject_name);
    subject_manual_joint_path = strcat(MANUAL_JOINT_TRACKS_PATH,'/',subject_name);
    
    list_of_phases = dir(fullfile(subject_path,'/PHASE_A*'));
    
    % Iterating through each phase
    for l = 1:1:length(list_of_phases)
        phase_name = list_of_phases(l).name;
        phase_video_path = sprintf('%s/%s',subject_path,phase_name);
        phase_optflow_path = sprintf('%s/%s',subject_optflow_path,phase_name);
        
        %        phase_fg_video_path = sprintf('%s/%s',subject_fg_path,phase_name);'
        phase_manual_joint_path = strcat(subject_manual_joint_path,'/',phase_name);
      
       list_of_video_files = dir(fullfile(phase_video_path,'/*.avi'));
       list_of_files_manual = dir(fullfile(phase_manual_joint_path,'/*.txt'));
       
       % Iterating through each file
       for m = 1:1:length(list_of_video_files) % for each version number of files are the same
           
           video_file = sprintf('%s/%s',phase_video_path,list_of_video_files(m).name);
           obj_video = VideoReader(video_file);
           frames = read(obj_video,[1 inf]);
           
%            fg_video_file = sprintf('%s/%s',phase_fg_video_path,list_of_video_files(m).name);
%            obj_video_fg = VideoReader(fg_video_file);
%            frames_fg = read(obj_video_fg);
           
           num_frames = size(frames,4);
           num_rows = size(frames,1);
           num_cols = size(frames,2);
           num_bands = size(frames,3);
           
           file_name_manual = list_of_files_manual(m).name;
           
           dpm_detector_file = sprintf('%s/Detections_%s',phase_video_path,file_name_manual);
           
           %art_detector_file = sprintf('%s/ArtDet_%s',phase_video_path,file_name_manual);
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
           
           %% WRITE CODE HERE TO EXTRACT THE OPTICAL FLOW MAGNITUDE AND 
           %% AND DIRECTION, THEN USE HHOF ON THE DETECTED LOCATION AS THE MOTION FEATURE VECTOR
           % Load optical flow magnitude and direction          
           % opening the articulated model detector file
%           fid = fopen(art_detector_file,'w');
           fid = fopen(motion_descriptor_file,'w');
              
           start_frame = A(1,1);
           num_of_frames = size(A,1);

%            total_detected_time = 0;
%            
%            num_parts = 26;
%            
%            %parts_idx = 1:1:num_of_parts;
%            parts_idx = [3 12 4 16 6 18 10 22 12 24 14 26];
%            idxs_parts = reshape([2*parts_idx-1 ; 2*parts_idx],1,2*length(parts_idx));
%            % creating the set of frame list and the set of frames for
%            % running the bin detector
%            X_det_prev = zeros(1,num_parts*2);
           for fr_num = start_frame:1:start_frame + num_of_frames-1                
               
               % load the image
               img = frames(:,:,:,fr_num);
               
               % load the optical flow
               optMag_file = sprintf('%s/optFlow_mag_framecount_%d.txt',phase_optflow_path,fr_num);
               optDir_file = sprintf('%s/optFlow_dir_framecount_%d.txt',phase_optflow_path,fr_num);
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
               
               try
                    crop_size = [A(fr_num - start_frame + 1,2:5)];
               catch exception
                   disp(exception)
               end
                  
               % crop the image and the optical flow
               im = imcrop(img,crop_size);
               im_mag = imcrop(img_mag,crop_size);
               im_dir = imcrop(img_dir,crop_size);
               
               % Compute HHOF vector
               
               
               % save HHOF vector
               
               
%                im = imresize(im,2,'bicubic');
% 
%                % performing the articulated model detection
%                tic;
%                boxes = detect(im, model, min(model.thresh,-1));
%                dettime = toc; % record cpu time
%                total_detected_time = dettime + total_detected_time;
%                
%                % This is the original method
%                boxes = nms(boxes); % nonmaximal suppression
               %colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
               
               % We can apply the BMVC method here
               
               
               %
               %showboxes(im, boxes(1,:),colorset); % show the best detection
               %showboxes(im, boxes,colorset);  % show all detections
               %fprintf('detection took %.1f seconds\n',dettime);
               
               
               % Another method to prune out detections other than NMS: BMVC 2013
               
%                % Store the one with the maximum score
%                if(isempty(boxes)) % if empty boxes
%                    % check if the previous value is zero
%                    % If not, then there were some frames detected earlier
%                    if( ~(sum(X_det_prev) == 0) ) 
%                        fprintf(fid,'%f\t',X_det_prev(1,idxs_parts));
%                        fprintf(fid,'\n');
%                    end
%                    continue;
%                else % if not empty boxes
%                    % if X_det_prev is zero, then this must be the first
%                    % frame that had some detections
%                    if(sum(X_det_prev) == 0)
%                        fprintf(fid,'%d\n',start_frame + num_of_frames - fr_num);
%                        fprintf(fid,'%d\n',length(parts_idx));
%                        fprintf(fid,'%d\n',fr_num);
%                    end
%                end
%                for b_num = 1:1:size(boxes(1,:),1)
%                    %fprintf(fid,'%d\t',fr_num);
%                    %for bb = 1:1:size(boxes(b_num,:),2)
%                        box_det = boxes(b_num,:);
%                        
%                        % get the original resolution back
%                        box_det(:,1:end-2) = box_det(:,1:end-2) / 2;
%                        
%                        % shift the coordinates to get absolute coordinates
%                        box_det(:,1:2:end-2) = box_det(:,1:2:end-2) + crop_size(1);
%                        box_det(:,2:2:end-2) = box_det(:,2:2:end-2) + crop_size(2);
%                        
%                        % convert to center coordinates
%                        numpart = floor(size(box_det,2)/4);
%                        x1_det = zeros(size(box_det,1),numpart);
%                        y1_det = zeros(size(box_det,1),numpart);
%                        x2_det = zeros(size(box_det,1),numpart);
%                        y2_det = zeros(size(box_det,1),numpart);
%                        
%                        for p = 1:numpart
%                          x1_det(:,p) = box_det(:,1+(p-1)*4);
%                          y1_det(:,p) = box_det(:,2+(p-1)*4);
%                          x2_det(:,p) = box_det(:,3+(p-1)*4);
%                          y2_det(:,p) = box_det(:,4+(p-1)*4);
%                          
%                          cent_x(:,p) = x1_det(:,p) + (x2_det(:,p) - x1_det(:,p))/2;
%                          cent_y(:,p) = y1_det(:,p) + (y2_det(:,p) - y1_det(:,p))/2;
%                        end
%                        
%                        X = zeros(size(box_det,1),numpart*2);
%                        for p = 1:1:numpart
%                            X(:,2*p-1) = cent_x(:,p); % X-coordinate in odd locations
%                            X(:,2*p) = cent_y(:,p); % Y-coordinate in even locations
%                        end
%                        X_det_prev = X;
%                        fprintf(fid,'%f\t',X(:,idxs_parts));
%                    %end
%                    %fprintf(fid,'%f\n',dettime);
%                    fprintf(fid,'\n');
%                end
%                
            end
            fclose(fid);
           
           Filenames = [Filenames ; {file_name_manual(1:length(file_name_manual)-10)}];
           Start_frames = [Start_frames ; start_frame];
           
           s_mesg = sprintf('%s processed in %f secs',dpm_detector_file,total_detected_time);
           disp(s_mesg);
           
       end
       
       
    end
    
end
