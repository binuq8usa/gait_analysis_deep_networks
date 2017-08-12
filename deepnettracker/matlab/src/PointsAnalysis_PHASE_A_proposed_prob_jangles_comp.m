%% Script file to load the tracked joint locations of all sequences belonging to all the phases
% Setting the path variables 
HOME_PATH = '~';
DATASET_PATH = '/Datasets/AFITUnclassifiedInformation';
TRACKED_JOINT_TRACKS_PATH = strcat(HOME_PATH,DATASET_PATH,'/JointTracks');
MANUAL_JOINT_TRACKS_PATH = strcat(HOME_PATH,DATASET_PATH);
TRACKED_JOINT_ANGLE_TRACKS_PATH = strcat(HOME_PATH,DATASET_PATH,'/JointAngleTracks');
Joint_names = {'Shoulder';'Elbow';'Wrist';'Hip';'Knee';'Ankle'};

% creating the directory to store joint angle trajectories
status = mkdir(TRACKED_JOINT_ANGLE_TRACKS_PATH);

jointAngleNames = {'Elbow Angle';'Shoulder Angle Pos'};
jointAngleNames_Comp = {'Elbow Angle PLS';'Shoulder Angle Pos PLS'; 'Elbow Angle SURF';'Shoulder Angle Pos SURF'};
jointAngleNames_inFilename = {'WristAngle';'ShoulderAnglePos'};

% Getting the list of subjects
list_of_subjects = dir(fullfile(TRACKED_JOINT_TRACKS_PATH,'/Subject*'));

% Manual points
Acc_Manual_Pts = [];

% Tracked point with 7 different descriptors using the proposed method
Acc_Tracked_Pts_hog_Proposed_prob = [];
Acc_Tracked_Pts_lbp_Proposed_prob = [];
Acc_Tracked_Pts_sift_Proposed_prob = [];
Acc_Tracked_Pts_surf_Proposed_prob = [];
Acc_Tracked_Pts_brief_Proposed_prob = [];
Acc_Tracked_Pts_brisk_Proposed_prob = [];
Acc_Tracked_Pts_orb_Proposed_prob = [];

Start_frames = [];

% Distance measures for each sequence for each descriptor in the proposed
% method
Dist_Joints_hog_Proposed_prob = [];
Dist_Joints_lbp_Proposed_prob = [];
Dist_Joints_sift_Proposed_prob = [];
Dist_Joints_surf_Proposed_prob = [];
Dist_Joints_brief_Proposed_prob = [];
Dist_Joints_brisk_Proposed_prob = [];
Dist_Joints_orb_Proposed_prob = [];

Filenames = [];

TrackingMetrics_key = {'MOTP'; 'MOTA';'rateFP';'rateFN';'rateTP'};
OtherMetrics_key = {'TP','FN','FP','IDSW','gt'};

TrackingMetrics_hog_Proposed_prob = [];
TrackingMetrics_lbp_Proposed_prob = [];
TrackingMetrics_sift_Proposed_prob = [];
TrackingMetrics_surf_Proposed_prob = [];
TrackingMetrics_brief_Proposed_prob = [];
TrackingMetrics_brisk_Proposed_prob = [];
TrackingMetrics_orb_Proposed_prob = [];
% 
TrackingMetrics_hog_Proposed_prob_n = [];
TrackingMetrics_lbp_Proposed_prob_n = [];
TrackingMetrics_sift_Proposed_prob_n = [];
TrackingMetrics_surf_Proposed_prob_n = [];
TrackingMetrics_brief_Proposed_prob_n = [];
TrackingMetrics_brisk_Proposed_prob_n = [];
TrackingMetrics_orb_Proposed_prob_n = [];

OtherMetrics_hog_Proposed_prob = [];
OtherMetrics_lbp_Proposed_prob = [];
OtherMetrics_sift_Proposed_prob = [];
OtherMetrics_surf_Proposed_prob = [];
OtherMetrics_brief_Proposed_prob = [];
OtherMetrics_brisk_Proposed_prob = [];
OtherMetrics_orb_Proposed_prob = [];

OtherMetrics_hog_Proposed_prob_n = [];
OtherMetrics_lbp_Proposed_prob_n = [];
OtherMetrics_sift_Proposed_prob_n = [];
OtherMetrics_surf_Proposed_prob_n = [];
OtherMetrics_brief_Proposed_prob_n = [];
OtherMetrics_brisk_Proposed_prob_n = [];
OtherMetrics_orb_Proposed_prob_n = [];

class_labels_vest_novest = [];

P_desc_hog_Proposed_prob = [];
P_desc_lbp_Proposed_prob = [];
P_desc_sift_Proposed_prob = [];
P_desc_surf_Proposed_prob = [];
P_desc_brief_Proposed_prob = [];
P_desc_brisk_Proposed_prob = [];
P_desc_orb_Proposed_prob = [];
P_desc_m = [];

% Iterating through each subject
for k = 1:1:length(list_of_subjects)
    subject_name = list_of_subjects(k).name;
    subject_tracked_joint_path = strcat(TRACKED_JOINT_TRACKS_PATH,'/',subject_name);
    subject_manual_joint_path = strcat(MANUAL_JOINT_TRACKS_PATH,'/',subject_name);
    
    subject_tracked_joint_angle_path = strcat(TRACKED_JOINT_ANGLE_TRACKS_PATH,'/',subject_name);
    status = mkdir(subject_tracked_joint_angle_path);
    
    list_of_phases = dir(fullfile(subject_tracked_joint_path,'/PHASE_A*'));
    
    % Iterating through each phase
    for l = 1:1:length(list_of_phases)
       phase_name = list_of_phases(l).name;
       phase_tracked_joint_path = strcat(subject_tracked_joint_path,'/',phase_name);
       phase_manual_joint_path = strcat(subject_manual_joint_path,'/',phase_name);
       
       phase_tracked_joint_angle_path = strcat(subject_tracked_joint_angle_path,'/',phase_name);
       status = mkdir(phase_tracked_joint_angle_path);
       
       list_of_files_hog_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_hog_Proposed_prob.txt'));
       list_of_files_lbp_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_lbp_Proposed_prob.txt'));
       list_of_files_sift_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_sift_Proposed_prob.txt'));
       list_of_files_surf_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_surf_Proposed_prob.txt'));
       list_of_files_brief_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_brief_Proposed_prob.txt'));
       list_of_files_brisk_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_brisk_Proposed_prob.txt'));
       list_of_files_orb_Proposed_prob = dir(fullfile(phase_tracked_joint_path,'/*_orb_Proposed_prob.txt'));
       
       list_of_files_manual = dir(fullfile(phase_manual_joint_path,'/*.txt'));
       
       % Iterating through each file
       for m = 1:1:length(list_of_files_sift_Proposed_prob) % for each version number of files are the same
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
               Dist_hog_Proposed_prob = [];
               TrackedPts_hog_Proposed_prob_new = [];

               % Load the tracked points for version 2
               fid_2 = fopen(tracked_pts_filename_lbp_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_lbp_Proposed_prob = A';
               fclose(fid_2);
               Dist_lbp_Proposed_prob = [];
               TrackedPts_lbp_Proposed_prob_new = [];

               % Load the tracked points for version 3
               fid_2 = fopen(tracked_pts_filename_sift_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_sift_Proposed_prob = A';
               fclose(fid_2);
               Dist_sift_Proposed_prob = [];
               TrackedPts_sift_Proposed_prob_new = [];

               % Load the tracked points for version 4
               fid_2 = fopen(tracked_pts_filename_surf_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_surf_Proposed_prob = A';
               fclose(fid_2);
               Dist_surf_Proposed_prob = [];
               TrackedPts_surf_Proposed_prob_new = [];

               % Load the tracked points for version 2
               fid_2 = fopen(tracked_pts_filename_brief_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_brief_Proposed_prob = A';
               fclose(fid_2);
               Dist_brief_Proposed_prob = [];
               TrackedPts_brief_Proposed_prob_new = [];

               % Load the tracked points for version 3
               fid_2 = fopen(tracked_pts_filename_brisk_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_brisk_Proposed_prob = A';
               fclose(fid_2);
               Dist_brisk_Proposed_prob = [];
               TrackedPts_brisk_Proposed_prob_new = [];

               % Load the tracked points for version 4
               fid_2 = fopen(tracked_pts_filename_orb_Proposed_prob);
               A = fscanf(fid_2,'%f',[2*num_of_joints, num_of_frames]);
               if(size(A,1)==0)
                   continue
               end
               TrackedPts_orb_Proposed_prob = A';
               fclose(fid_2);
               Dist_orb_Proposed_prob = [];
               TrackedPts_orb_Proposed_prob_new = [];
               
               p_desc_hog_Proposed_prob = [];
               p_desc_lbp_Proposed_prob = [];
               p_desc_sift_Proposed_prob = [];
               p_desc_surf_Proposed_prob = [];
               p_desc_brief_Proposed_prob = [];
               p_desc_brisk_Proposed_prob = [];
               p_desc_orb_Proposed_prob = [];
                
               p_desc_m = [];
               
           catch exception
               disp(exception)
               continue
           end
           
           jointColor = {'b','r','g','c','y','m'};
           
           % computing the joint angles
           joint_angle_traj_hog_Proposed_prob = ComputeJointAngles(TrackedPts_hog_Proposed_prob,720,480);
           p_desc_hog_Proposed_prob = [p_desc_hog_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_hog_Proposed_prob,win_size)];
           
           joint_angle_traj_lbp_Proposed_prob = ComputeJointAngles(TrackedPts_lbp_Proposed_prob,720,480);
           p_desc_lbp_Proposed_prob = [p_desc_lbp_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_lbp_Proposed_prob,win_size)];
           
           joint_angle_traj_sift_Proposed_prob = ComputeJointAngles(TrackedPts_sift_Proposed_prob,720,480);
           p_desc_sift_Proposed_prob = [p_desc_sift_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_sift_Proposed_prob,win_size)];
           
           joint_angle_traj_surf_Proposed_prob = ComputeJointAngles(TrackedPts_surf_Proposed_prob,720,480);
           p_desc_surf_Proposed_prob = [p_desc_surf_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_surf_Proposed_prob,win_size)];
           
           joint_angle_traj_brief_Proposed_prob = ComputeJointAngles(TrackedPts_brief_Proposed_prob,720,480);
           p_desc_brief_Proposed_prob = [p_desc_brief_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_brief_Proposed_prob,win_size)];
           
           joint_angle_traj_brisk_Proposed_prob = ComputeJointAngles(TrackedPts_brisk_Proposed_prob,720,480);
           p_desc_brisk_Proposed_prob = [p_desc_brisk_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_brisk_Proposed_prob,win_size)];
           
           joint_angle_traj_orb_Proposed_prob = ComputeJointAngles(TrackedPts_orb_Proposed_prob,720,480);
           p_desc_orb_Proposed_prob = [p_desc_orb_Proposed_prob ComputeShapeOfTrajectory(joint_angle_traj_orb_Proposed_prob,win_size)];
           
           joint_angle_traj_man_Proposed_prob = ComputeJointAngles(ManualPts(2:end,:),720,480);
           p_desc_m = [p_desc_m ComputeShapeOfTrajectory(joint_angle_traj_man_Proposed_prob,win_size)];
                 
           % Plot the joint angle trajectories using each descriptor
           % HOG
           num_of_angles = size(joint_angle_traj_man_Proposed_prob,2);
           figure(1);
           for jo = 1:1:num_of_angles
               %plot(X_hog_Proposed_prob_new,Y_hog_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
               plot(2:1:num_of_frames,smooth(joint_angle_traj_hog_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from HOG');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_hog_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % LBP
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_lbp_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from LBP');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_lbp_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % SIFT
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_sift_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from SIFT');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_sift_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % SURF
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_surf_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from SURF');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_surf_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % BRIEF
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_brief_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from BRIEF');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_brief_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % BRISK
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_brisk_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from BRISK');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_brisk_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % ORB
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_orb_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories obtained from ORB');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_orb_Proposed_prob(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % Manual
           figure(1);
           for jo = 1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories using Manual Points');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj.jpg',phase_tracked_joint_angle_path,file_name_manual(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;
           
           % comparison with the best tracking scheme
           figure(1);
           for jo=1:1:num_of_angles
               col_joint = sprintf('*%s--',jointColor{jo});
               plot(2:1:num_of_frames,smooth(joint_angle_traj_man_Proposed_prob(:,jo)),col_joint,'LineWidth',1);
               hold on;
           end
           for jo=1:1:num_of_angles
               plot(2:1:num_of_frames,smooth(joint_angle_traj_surf_Proposed_prob(:,jo)),jointColor{jo},'LineWidth',1.5);
               hold on;
           end
           legend(jointAngleNames_Comp);
           xlabel('Frame Number','FontSize',10,'FontWeight','bold');
           ylabel('Angle in Radians','FontSize',10,'FontWeight','bold');
           s_title = sprintf('Joint Angle Trajectories using Manual Points');
           title(s_title,'FontSize',10,'FontWeight','bold');
           set(gca,'FontSize',10,'FontWeight','bold');
           hold off
           grid on
           plot_filename = sprintf('%s/%s_JointAngleTraj_Comp.jpg',phase_tracked_joint_angle_path,file_name_manual(1:end-4));
           saveas(gcf,plot_filename,'jpg');
           close all;

           % TODO: Draw the figure for each subject and save figure
           for jo = 1:1:num_of_joints
               X_hog_Proposed_prob = TrackedPts_hog_Proposed_prob(:,2*jo-1);
               Y_hog_Proposed_prob = TrackedPts_hog_Proposed_prob(:,2*jo);
%                p_desc_hog_Proposed_prob = [p_desc_hog_Proposed_prob ComputeShapeOfTrajectory([X_hog_Proposed_prob Y_hog_Proposed_prob],win_size)];
% 
               X_lbp_Proposed_prob = TrackedPts_lbp_Proposed_prob(:,2*jo-1);
               Y_lbp_Proposed_prob = TrackedPts_lbp_Proposed_prob(:,2*jo);
%                p_desc_lbp_Proposed_prob = [p_desc_lbp_Proposed_prob ComputeShapeOfTrajectory([X_lbp_Proposed_prob Y_lbp_Proposed_prob],win_size)];
%        
               X_sift_Proposed_prob = TrackedPts_sift_Proposed_prob(:,2*jo-1);
               Y_sift_Proposed_prob = TrackedPts_sift_Proposed_prob(:,2*jo);
%                p_desc_sift_Proposed_prob = [p_desc_sift_Proposed_prob ComputeShapeOfTrajectory([X_sift_Proposed_prob Y_sift_Proposed_prob],win_size)];
%                
               X_surf_Proposed_prob = TrackedPts_surf_Proposed_prob(:,2*jo-1);
               Y_surf_Proposed_prob = TrackedPts_surf_Proposed_prob(:,2*jo);
%                p_desc_surf_Proposed_prob = [p_desc_surf_Proposed_prob ComputeShapeOfTrajectory([X_surf_Proposed_prob Y_surf_Proposed_prob],win_size)];
%  
               X_brief_Proposed_prob = TrackedPts_brief_Proposed_prob(:,2*jo-1);
               Y_brief_Proposed_prob = TrackedPts_brief_Proposed_prob(:,2*jo);
%                p_desc_brief_Proposed_prob = [p_desc_brief_Proposed_prob ComputeShapeOfTrajectory([X_brief_Proposed_prob Y_brief_Proposed_prob],win_size)];
%                
               X_brisk_Proposed_prob = TrackedPts_brisk_Proposed_prob(:,2*jo-1);
               Y_brisk_Proposed_prob = TrackedPts_brisk_Proposed_prob(:,2*jo);
%                p_desc_brisk_Proposed_prob = [p_desc_brisk_Proposed_prob ComputeShapeOfTrajectory([X_brisk_Proposed_prob Y_brisk_Proposed_prob],win_size)];
% 
               X_orb_Proposed_prob = TrackedPts_orb_Proposed_prob(:,2*jo-1);
               Y_orb_Proposed_prob = TrackedPts_orb_Proposed_prob(:,2*jo);
%                p_desc_orb_Proposed_prob = [p_desc_orb_Proposed_prob ComputeShapeOfTrajectory([X_orb_Proposed_prob Y_orb_Proposed_prob],win_size)];
 
               X_m = ManualPts(:,2*jo-1);
               Y_m = ManualPts(:,2*jo);
               net_m = newgrnn(X_m',Y_m');
               X_m_new = X_m(1):0.1:X_m(length(X_m));
               Y_m_new = sim(net_m,X_m_new);
               T = [1:1:num_of_frames-1]';
               
               %p_desc_m = [p_desc_m ComputeShapeOfTrajectory([X_m(2:num_of_frames,:) Y_m(2:num_of_frames,:)],win_size);];
               
               % Computing the covariance matrix  :  can be considered as a
               % kernel
               % Version 1
               len = length(X_hog_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_hog_Proposed_prob_n = X_hog_Proposed_prob - repmat(X_m(1),len,1);
               Y_hog_Proposed_prob_n = Y_hog_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_hog_Proposed_prob_n Y_hog_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_hog_Proposed_prob = [Dist_hog_Proposed_prob ; dist];
               
               % Version 2
               len = length(X_lbp_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_lbp_Proposed_prob_n = X_lbp_Proposed_prob - repmat(X_m(1),len,1);
               Y_lbp_Proposed_prob_n = Y_lbp_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_lbp_Proposed_prob_n Y_lbp_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_lbp_Proposed_prob = [Dist_lbp_Proposed_prob ; dist];
               
               % Version 3
               len = length(X_sift_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_sift_Proposed_prob_n = X_sift_Proposed_prob - repmat(X_m(1),len,1);
               Y_sift_Proposed_prob_n = Y_sift_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_sift_Proposed_prob_n Y_sift_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_sift_Proposed_prob = [Dist_sift_Proposed_prob ; dist];
               
               % Version 4
               len = length(X_surf_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_surf_Proposed_prob_n = X_surf_Proposed_prob - repmat(X_m(1),len,1);
               Y_surf_Proposed_prob_n = Y_surf_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_surf_Proposed_prob_n Y_surf_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_surf_Proposed_prob = [Dist_surf_Proposed_prob ; dist];
               
               % Version 5
               len = length(X_brief_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_brief_Proposed_prob_n = X_brief_Proposed_prob - repmat(X_m(1),len,1);
               Y_brief_Proposed_prob_n = Y_brief_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_brief_Proposed_prob_n Y_brief_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_brief_Proposed_prob = [Dist_brief_Proposed_prob ; dist];
               
               % Version 6
               len = length(X_brisk_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_brisk_Proposed_prob_n = X_brisk_Proposed_prob - repmat(X_m(1),len,1);
               Y_brisk_Proposed_prob_n = Y_brisk_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_brisk_Proposed_prob_n Y_brisk_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_brisk_Proposed_prob = [Dist_brisk_Proposed_prob ; dist];
               
               % Version 7
               len = length(X_orb_Proposed_prob);
               X_m_n = X_m(1:len) - repmat(X_m(1),len,1); % normalization with respect to the first point
               Y_m_n = Y_m(1:len) - repmat(Y_m(1),len,1);
               T = [1:1:len]';
               K_n = cov([X_m_n Y_m_n T]); % computing the covariance
               X_orb_Proposed_prob_n = X_orb_Proposed_prob - repmat(X_m(1),len,1);
               Y_orb_Proposed_prob_n = Y_orb_Proposed_prob - repmat(Y_m(1),len,1);
               K = cov([X_orb_Proposed_prob_n Y_orb_Proposed_prob_n T]);
               % comparing the covariance matrices using the metric
               lambda = eig(K,K_n);
               dist = sqrt(sum(log(lambda).*log(lambda)));
               Dist_orb_Proposed_prob = [Dist_orb_Proposed_prob ; dist];
           end
           
           %% TODO: Write up code to plot each joint angle trajectory for all subjects for both the manual and the best tracking scheme with demarcation of both the loaded and unloaded case
           % This is necessary to see if the trajectories vary with load
           % and no load case
           % Also, this can explore the possibility of extracting better
           % joint trajectory descriptors than the displacement vector
           % maybe, something to do with sinusoidal curve fitting where the
           % non-linear parameters are the features. 
           % In future, there will be more joint angle trajectories with
           % the discrete parts model.
           
%            % Plotting and saving the figure for each descriptor
%            % HOG
%            figure(1);
%            for jo = 1:1:num_joints
%                plot(X_hog_Proposed_prob_new,Y_hog_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_hog_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % LBP
%            for jo = 1:1:num_joints
%                plot(X_lbp_Proposed_prob_new,Y_lbp_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_lbp_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % SIFT
%            for jo = 1:1:num_joints
%                plot(X_sift_Proposed_prob_new,Y_sift_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_sift_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % SURF
%            for jo = 1:1:num_joints
%                plot(X_surf_Proposed_prob_new,Y_surf_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_surf_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % BRIEF
%            for jo = 1:1:num_joints
%                plot(X_brief_Proposed_prob_new,Y_brief_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_brief_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % BRISK
%            for jo = 1:1:num_joints
%                plot(X_brisk_Proposed_prob_new,Y_brisk_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_brisk_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
%            
%            % ORB
%            for jo = 1:1:num_joints
%                plot(X_orb_Proposed_prob_new,Y_orb_Proposed_prob_new,jointColor{jo},'LineWidth',1.5);
%                hold on;
%                legend('Shoulder','Elbow','Wrist','Hip','Knee','Ankle');
%                xlabel('X-Coordinate','FontSize',9,'FontWeight','bold');
%                ylabel('Y-Coordinate','FontSize',9,'FontWeight','bold');
%                s_title = sprintf('%s Joint',Joint_names{jo});
%                title(s_title,'FontSize',9,'FontWeight','bold');
%                set(gca,'FontSize',9,'FontWeight','bold');
%                hold off
%                plot_filename = sprintf('%s/%s.jpg',phase_tracked_joint_path,file_name_orb_Proposed_prob(1:end-4));
%                saveas(gcf,plot_filename,'jpg');
%                close all;
%            end
           
           % Generaring the data for evaluation for each feature descriptor
           % before the spatial linking neural network
           [gt_hog_Proposed_prob,results_hog_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_hog_Proposed_prob,num_of_joints);
           [gt_lbp_Proposed_prob,results_lbp_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_lbp_Proposed_prob,num_of_joints);
           [gt_sift_Proposed_prob,results_sift_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_sift_Proposed_prob,num_of_joints);
           [gt_surf_Proposed_prob,results_surf_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_surf_Proposed_prob,num_of_joints);
           [gt_brief_Proposed_prob,results_brief_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_brief_Proposed_prob,num_of_joints);
           [gt_brisk_Proposed_prob,results_brisk_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_brisk_Proposed_prob,num_of_joints);
           [gt_orb_Proposed_prob,results_orb_Proposed_prob] = generateData_JointTracks_wacv(ManualPts,TrackedPts_orb_Proposed_prob,num_of_joints);
           
%            % Generaring the data for evaluation for each feature descriptor
%            % after the spatial linking neural network ( its should be
%            '_new' not '_n'
%            [gt_hog_Proposed_prob_n,results_hog_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_hog_Proposed_prob_n);
%            [gt_lbp_Proposed_prob_n,results_lbp_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_lbp_Proposed_prob_n);
%            [gt_sift_Proposed_prob_n,results_sift_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_sift_Proposed_prob_n);
%            [gt_surf_Proposed_prob_n,results_surf_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_surf_Proposed_prob_n);
%            [gt_brief_Proposed_prob_n,results_brief_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_brief_Proposed_prob_n);
%            [gt_brisk_Proposed_prob_n,results_brisk_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_brisk_Proposed_prob_n);
%            [gt_orb_Proposed_prob_n,results_orb_Proposed_prob_n] = generateData_JointTracks_wacv(ManualPts,TrackedPts_orb_Proposed_prob_n);
           
           % Evaluating the ClearMOT metrics for each feature descriptor :
           % Without GRNN
           ClearMOT = evaluateMOT(gt_hog_Proposed_prob,results_hog_Proposed_prob,0.5,false);
           TrackingMetrics_hog_Proposed_prob = [TrackingMetrics_hog_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_hog_Proposed_prob = [OtherMetrics_hog_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];

           ClearMOT = evaluateMOT(gt_lbp_Proposed_prob,results_lbp_Proposed_prob,0.5,false);
           TrackingMetrics_lbp_Proposed_prob = [TrackingMetrics_lbp_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_lbp_Proposed_prob = [OtherMetrics_lbp_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           ClearMOT = evaluateMOT(gt_sift_Proposed_prob,results_sift_Proposed_prob,0.5,false);
           TrackingMetrics_sift_Proposed_prob = [TrackingMetrics_sift_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_sift_Proposed_prob = [OtherMetrics_sift_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           ClearMOT = evaluateMOT(gt_surf_Proposed_prob,results_surf_Proposed_prob,0.5,false);
           TrackingMetrics_surf_Proposed_prob = [TrackingMetrics_surf_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_surf_Proposed_prob = [OtherMetrics_surf_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           ClearMOT = evaluateMOT(gt_brief_Proposed_prob,results_brief_Proposed_prob,0.5,false);
           TrackingMetrics_brief_Proposed_prob = [TrackingMetrics_brief_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_brief_Proposed_prob = [OtherMetrics_brief_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           ClearMOT = evaluateMOT(gt_brisk_Proposed_prob,results_brisk_Proposed_prob,0.5,false);
           TrackingMetrics_brisk_Proposed_prob = [TrackingMetrics_brisk_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_brisk_Proposed_prob = [OtherMetrics_brisk_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           ClearMOT = evaluateMOT(gt_orb_Proposed_prob,results_orb_Proposed_prob,0.5,false);
           TrackingMetrics_orb_Proposed_prob = [TrackingMetrics_orb_Proposed_prob [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
           OtherMetrics_orb_Proposed_prob = [OtherMetrics_orb_Proposed_prob [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
%            % Evaluating the ClearMOT metrics for each feature descriptor :
%            % With GRNN spatio-temporal linking
%            ClearMOT = evaluateMOT(gt_hog_Proposed_prob_n,results_hog_Proposed_prob_n,0.5,false);
%            TrackingMetrics_hog_Proposed_prob_n = [TrackingMetrics_hog_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_hog_Proposed_prob_n = [OtherMetrics_hog_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_lbp_Proposed_prob_n,results_lbp_Proposed_prob_n,0.5,false);
%            TrackingMetrics_lbp_Proposed_prob_n = [TrackingMetrics_lbp_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_lbp_Proposed_prob_n = [OtherMetrics_lbp_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_sift_Proposed_prob_n,results_sift_Proposed_prob_n,0.5,false);
%            TrackingMetrics_sift_Proposed_prob_n = [TrackingMetrics_sift_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_sift_Proposed_prob_n = [OtherMetrics_sift_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_surf_Proposed_prob_n,results_surf_Proposed_prob_n,0.5,false);
%            TrackingMetrics_surf_Proposed_prob_n = [TrackingMetrics_surf_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_surf_Proposed_prob_n = [OtherMetrics_surf_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_brief_Proposed_prob_n,results_brief_Proposed_prob_n,0.5,false);
%            TrackingMetrics_brief_Proposed_prob_n = [TrackingMetrics_brief_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_brief_Proposed_prob_n = [OtherMetrics_brief_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_brisk_Proposed_prob_n,results_brisk_Proposed_prob_n,0.5,false);
%            TrackingMetrics_brisk_Proposed_prob_n = [TrackingMetrics_brisk_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_brisk_Proposed_prob_n = [OtherMetrics_brisk_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
%            
%            ClearMOT = evaluateMOT(gt_orb_Proposed_prob_n,results_orb_Proposed_prob_n,0.5,false);
%            TrackingMetrics_orb_Proposed_prob_n = [TrackingMetrics_orb_Proposed_prob_n [ClearMOT.MOTP ; ClearMOT.MOTA ; ClearMOT.rateFP; ClearMOT.rateFN ; ClearMOT.rateTP] ];
%            OtherMetrics_orb_Proposed_prob_n = [OtherMetrics_orb_Proposed_prob_n [ClearMOT.TP ; ClearMOT.FN ; ClearMOT.FP ; ClearMOT.IDSW; ClearMOT.gt] ];
           
           class_labels_vest_novest = [class_labels_vest_novest ; m * ones(size(p_desc_m,1),1)];
           P_desc_hog_Proposed_prob = [P_desc_hog_Proposed_prob ; p_desc_hog_Proposed_prob];
           P_desc_lbp_Proposed_prob = [P_desc_lbp_Proposed_prob ; p_desc_lbp_Proposed_prob];
           P_desc_sift_Proposed_prob = [P_desc_sift_Proposed_prob ; p_desc_sift_Proposed_prob];
           P_desc_surf_Proposed_prob = [P_desc_surf_Proposed_prob ; p_desc_surf_Proposed_prob];
           P_desc_brief_Proposed_prob = [P_desc_brief_Proposed_prob ; p_desc_brief_Proposed_prob];
           P_desc_brisk_Proposed_prob = [P_desc_brisk_Proposed_prob ; p_desc_brisk_Proposed_prob];
           P_desc_orb_Proposed_prob = [P_desc_orb_Proposed_prob ; p_desc_orb_Proposed_prob];
           P_desc_m = [P_desc_m ; p_desc_m];
           
           %% TODO: Normalize the x,y coordinates with respect to the first manual point
           norm_factor = ManualPts(1,:);
           ManualPts = ManualPts - repmat(norm_factor,num_of_frames,1);
           len = size(TrackedPts_hog_Proposed_prob,1);
           TrackedPts_hog_Proposed_prob = TrackedPts_hog_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_lbp_Proposed_prob,1);
           TrackedPts_lbp_Proposed_prob = TrackedPts_lbp_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_sift_Proposed_prob,1);
           TrackedPts_sift_Proposed_prob = TrackedPts_sift_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_surf_Proposed_prob,1);
           TrackedPts_surf_Proposed_prob = TrackedPts_surf_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_brief_Proposed_prob,1);
           TrackedPts_brief_Proposed_prob = TrackedPts_brief_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_brisk_Proposed_prob,1);
           TrackedPts_brisk_Proposed_prob = TrackedPts_brisk_Proposed_prob - repmat(norm_factor,len,1);
           len = size(TrackedPts_orb_Proposed_prob,1);
           TrackedPts_orb_Proposed_prob = TrackedPts_orb_Proposed_prob - repmat(norm_factor,len,1);
           
           % Store the tracks for each joint
           Acc_Manual_Pts = [Acc_Manual_Pts ; ManualPts];
           
           Acc_Tracked_Pts_hog_Proposed_prob = [Acc_Tracked_Pts_hog_Proposed_prob ; TrackedPts_hog_Proposed_prob];
           Acc_Tracked_Pts_lbp_Proposed_prob = [Acc_Tracked_Pts_lbp_Proposed_prob ; TrackedPts_lbp_Proposed_prob];
           Acc_Tracked_Pts_sift_Proposed_prob = [Acc_Tracked_Pts_sift_Proposed_prob ; TrackedPts_sift_Proposed_prob];
           Acc_Tracked_Pts_surf_Proposed_prob = [Acc_Tracked_Pts_surf_Proposed_prob ; TrackedPts_surf_Proposed_prob];
           Acc_Tracked_Pts_brief_Proposed_prob = [Acc_Tracked_Pts_brief_Proposed_prob ; TrackedPts_brief_Proposed_prob];
           Acc_Tracked_Pts_brisk_Proposed_prob = [Acc_Tracked_Pts_brisk_Proposed_prob ; TrackedPts_brisk_Proposed_prob];
           Acc_Tracked_Pts_orb_Proposed_prob = [Acc_Tracked_Pts_orb_Proposed_prob ; TrackedPts_orb_Proposed_prob];
           
           Start_frames = [Start_frames ; start_frame];
           
           Dist_Joints_hog_Proposed_prob = [Dist_Joints_hog_Proposed_prob Dist_hog_Proposed_prob];
           Dist_Joints_lbp_Proposed_prob = [Dist_Joints_lbp_Proposed_prob Dist_lbp_Proposed_prob];
           Dist_Joints_sift_Proposed_prob = [Dist_Joints_sift_Proposed_prob Dist_sift_Proposed_prob];
           Dist_Joints_surf_Proposed_prob = [Dist_Joints_surf_Proposed_prob Dist_surf_Proposed_prob];
           Dist_Joints_brief_Proposed_prob = [Dist_Joints_brief_Proposed_prob Dist_brief_Proposed_prob];
           Dist_Joints_brisk_Proposed_prob = [Dist_Joints_brisk_Proposed_prob Dist_brisk_Proposed_prob];
           Dist_Joints_orb_Proposed_prob = [Dist_Joints_orb_Proposed_prob Dist_orb_Proposed_prob];
           
           Filenames = [Filenames ; {file_name_manual(1:length(file_name_manual)-10)}];
           
           fprintf('%s processed\n',file_name_manual);
           
       end
       
       
    end
    
end