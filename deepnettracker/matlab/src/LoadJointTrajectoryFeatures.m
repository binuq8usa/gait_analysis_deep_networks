%% Script file to load the tracked joint locations of all sequences belonging to all the phases and compute the joint angle 
%% TODO: LATER compute the motion flow and motion descriptors corresponding to each body joint and learn the relationship between the pose position and joint description
%% with the kinematic model (joint angle configuration)
%% this kinematic model is computed using the true anatomical information and the actual Groebner basis equations at different parts of the gait cycle.

%% NEED TO READ IN THE ACTUAL ANATOMICAL INFORMATION FOR JOINT ANGLES
%% ALSO, THE END EFFECTOR POSITIONS SHOULD BE FROM THE GROUND TRUTH DATA (PLS OR MANUAL POINTS)
%% The input pose should be the discrete parts model
clear all;
clc;
% Setting the path variables 
HOME_PATH = getenv('HOME'); 
GAITANALYSIS_PATH = '/Datasets/GaitAnalysis';
DATASET_PATH = sprintf('%s%s/AFITUnclassifiedInformation',HOME_PATH,GAITANALYSIS_PATH);
OPTICAL_FLOW_PATH = sprintf('%s%s/AFITUnclassifiedInformation/OpticalFlow',HOME_PATH,GAITANALYSIS_PATH);
TRACKED_JOINT_TRACKS_PATH = strcat(DATASET_PATH,'/JointTracks');
MANUAL_JOINT_TRACKS_PATH = DATASET_PATH;

% PATH TO MOTION FEATURE EXTRACTION

% Getting the list of subjects
list_of_subjects = dir(fullfile(TRACKED_JOINT_TRACKS_PATH,'/Subject*'));
Filenames = [];

data_angles = cell(length(list_of_subjects),5); % 5 phases
class_labels = cell(length(list_of_subjects),5);

data_angles_per_seq = cell(2*length(list_of_subjects),5); % 5 phases
class_labels_per_seq = cell(2*length(list_of_subjects),5);

% Iterating through each subject
for k = 1:1:length(list_of_subjects)
    subject_name = list_of_subjects(k).name;
    subject_tracked_joint_path = strcat(TRACKED_JOINT_TRACKS_PATH,'/',subject_name);
    subject_manual_joint_path = strcat(MANUAL_JOINT_TRACKS_PATH,'/',subject_name);
    subject_path = sprintf('%s/Deinterlaced/%s',DATASET_PATH,subject_name);
    
%     subject_tracked_joint_angle_path = strcat(TRACKED_JOINT_ANGLE_TRACKS_PATH,'/',subject_name);
%     status = mkdir(subject_tracked_joint_angle_path);
    
    list_of_phases = dir(fullfile(subject_tracked_joint_path,'/PHASE_*'));
    
    % Iterating through each phase
    for l = 1:1:length(list_of_phases)
        
        % accumulating the angles for all sequences for every subject of
        % every phase
       class_labels_vest_novest = [];
       joint_angle_ref = [];
       pos_ref = [];
       HH_video_ref = [];
       
       phase_name = list_of_phases(l).name;
       if(strcmp(phase_name,'PHASE_D'))
           continue;
       end
       phase_tracked_joint_path = strcat(subject_tracked_joint_path,'/',phase_name);
       phase_manual_joint_path = strcat(subject_manual_joint_path,'/',phase_name);
       phase_video_path = sprintf('%s/%s',subject_path,phase_name);
       
%        phase_tracked_joint_angle_path = strcat(subject_tracked_joint_angle_path,'/',phase_name);
%        status = mkdir(phase_tracked_joint_angle_path);
       
       list_of_files_hog_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_hog_Proposed_prob.txt'));
       list_of_files_lbp_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_lbp_Proposed_prob.txt'));
       list_of_files_sift_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_sift_Proposed_prob.txt'));
       list_of_files_surf_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_surf_Proposed_prob.txt'));
       list_of_files_brief_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_brief_Proposed_prob.txt'));
       list_of_files_brisk_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_brisk_Proposed_prob.txt'));
       list_of_files_orb_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_orb_Proposed_prob.txt'));
       
       list_of_files_manual = dir(fullfile(phase_manual_joint_path,'/*.txt'));
       
       % Iterating through each file
       num_of_seq_files_per_subject = length(list_of_files_sift_Proposed_prob);
       for m = 1:1:length(list_of_files_sift_Proposed_prob) % for each version number of files are the same
           
           joint_angle_ref_seq = [];
           pos_ref_seq = [];
           
           file_name_hog_Proposed_prob = list_of_files_hog_Proposed_prob(m).name;
           tracked_pts_filename_hog_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_hog_Proposed_prob);
           
           file_name_lbp_Proposed_prob = list_of_files_lbp_Proposed_prob(m).name;
           tracked_pts_filename_lbp_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_lbp_Proposed_prob);
           
           file_name_sift_Proposed_prob = list_of_files_sift_Proposed_prob(m).name;
           tracked_pts_filename_sift_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_sift_Proposed_prob);
           
           file_name_surf_Proposed_prob = list_of_files_surf_Proposed_prob(m).name;
           tracked_pts_filename_surf_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_surf_Proposed_prob);
           
           file_name_brief_Proposed_prob = list_of_files_brief_Proposed_prob(m).name;
           tracked_pts_filename_brief_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_brief_Proposed_prob);
           
           file_name_brisk_Proposed_prob = list_of_files_brisk_Proposed_prob(m).name;
           tracked_pts_filename_brisk_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_brisk_Proposed_prob);
           
           file_name_orb_Proposed_prob = list_of_files_orb_Proposed_prob(m).name;
           tracked_pts_filename_orb_Proposed_prob = strcat(phase_tracked_joint_path,'/',file_name_orb_Proposed_prob);
           
           file_name_manual = list_of_files_manual(m).name;
           manual_pts_filename = strcat(phase_manual_joint_path,'/',file_name_manual);
           
           motion_descriptor_file = sprintf('%s/motionDescriptor_%s',phase_video_path,file_name_manual);
           image_descriptor_file = sprintf('%s/imageDescriptor_%s',phase_video_path,file_name_manual);
           
           
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
           
           % Load the tracked points ; avoid any sequences where one of the
           % files is corrupted
           try
               % Load the tracked points for version 1
               fid_2 = fopen(tracked_pts_filename_hog_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_hog_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 2
               fid_2 = fopen(tracked_pts_filename_lbp_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_lbp_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 3
               fid_2 = fopen(tracked_pts_filename_sift_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_sift_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 4
               fid_2 = fopen(tracked_pts_filename_surf_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_surf_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 2
               fid_2 = fopen(tracked_pts_filename_brief_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_brief_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 3
               fid_2 = fopen(tracked_pts_filename_brisk_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_brisk_Proposed_prob = A';
               fclose(fid_2);

               % Load the tracked points for version 4
               fid_2 = fopen(tracked_pts_filename_orb_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_orb_Proposed_prob = A';
               fclose(fid_2); 
               
               fid_mm = fopen(motion_descriptor_file);
               A = fscanf(fid_mm,'%f',[4096, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               HH_video = A';
               fclose(fid_mm);
               
               fid_mm = fopen(image_descriptor_file);
               A = fscanf(fid_mm,'%f',[4096, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               HH_image = A';
               fclose(fid_mm);
               
           catch exception
               disp(exception)
               continue
           end
           
           % computing the joint angles
           [joint_angle_hog,pos_hog] = ComputeJointAngles(TrackedPts_hog_Proposed_prob,720,480);     
           [joint_angle_lbp,pos_lbp] = ComputeJointAngles(TrackedPts_lbp_Proposed_prob,720,480);    
           [joint_angle_sift,pos_sift] = ComputeJointAngles(TrackedPts_sift_Proposed_prob,720,480);
           [joint_angle_surf,pos_surf] = ComputeJointAngles(TrackedPts_surf_Proposed_prob,720,480);
           [joint_angle_brief,pos_brief] = ComputeJointAngles(TrackedPts_brief_Proposed_prob,720,480);
           [joint_angle_brisk,pos_brisk] = ComputeJointAngles(TrackedPts_brisk_Proposed_prob,720,480); 
           [joint_angle_orb,pos_orb] = ComputeJointAngles(TrackedPts_orb_Proposed_prob,720,480);
           [joint_angle_man,pos_man] = ComputeJointAngles(ManualPts(2:end,:),720,480);
           
           % compute the average value of the kinematic model vector using
           % different tracking schemes
           pos_avg = zeros(size(pos_hog));
           joint_angle_avg = zeros(size(joint_angle_hog));
           for kk = 1:1:size(pos_hog,1)
               pos_cum = [pos_hog(kk,:) ; pos_lbp(kk,:) ; pos_sift(kk,:) ; pos_surf(kk,:) ; pos_brief(kk,:) ; pos_brisk(kk,:) ; pos_orb(kk,:)];
               joint_angle_cum = [joint_angle_hog(kk,:) ; joint_angle_lbp(kk,:) ; joint_angle_sift(kk,:) ; joint_angle_surf(kk,:) ; joint_angle_brief(kk,:) ; joint_angle_brisk(kk,:) ; joint_angle_orb(kk,:)];
               
               pos_avg(kk,:) = median(pos_cum);
               joint_angle_avg(kk,:) = median(joint_angle_cum);
           end
           
                 
           class_labels_vest_novest = [class_labels_vest_novest ; m * ones(size(pos_sift,1),1)];
           
           joint_angle_ref = [joint_angle_ref ; joint_angle_avg]; % considering the tracking with surf descriptor
           pos_ref = [pos_ref ; pos_avg];
           
           joint_angle_ref_seq = [joint_angle_ref_seq ; joint_angle_avg]; % considering the tracking with surf descriptor
           pos_ref_seq = [pos_ref_seq ; pos_avg];
           
           HH_video_ref = [HH_video_ref ; HH_video];
           
           Filenames = [Filenames ; {file_name_manual(1:length(file_name_manual)-10)}];
           
           fprintf('%s processed\n',file_name_manual);
           
           data_angles_per_seq{num_of_seq_files_per_subject*(k-1)+m,l} = {joint_angle_ref_seq ; pos_ref_seq; m * ones(size(pos_avg,1),1); HH_video ; HH_image};
           
           
       end
       
       data_angles{k,l} = {joint_angle_ref ; pos_ref; class_labels_vest_novest};

    end
    
end