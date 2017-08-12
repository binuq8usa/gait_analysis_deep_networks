% function which takes in the tracked pose estimate locations and computes
% the corresponding joint angles using the Groebner basis equations
function [data_angles,data_pos] = ComputeJointAngles(data,x_max,y_max)

% the joint angles for the elbow and the wrist with shoulder as the origin
num_frames = size(data,1);
num_joints = size(data,2)/2;

%x_max = 720;
%y_max = 480;
% getting x and y coordinates with the image origin as the bottom left
% corner rather than the top left corner
x = zeros(num_frames,1);
y = zeros(num_frames,1);
for jj=1:1:num_joints
    x(:,jj) = data(:,2*jj-1);
    y(:,jj) = y_max - data(:,2*jj);
end

% compute parameters of the Grobner basis as given in page 14 of manuscript
a = x(:,3) - x(:,1); % normalized x-coordinate of the wrist joint with shoulder joint as origin
b = y(:,3) - y(:,1); % normalized y-coordinate of the wrist joint with shoulder joint as origin

L1 = sqrt( (x(:,2) - x(:,1)).^2 + (y(:,2) - y(:,1)).^2 ); % length of the upperarm
L3 = sqrt( (x(:,3) - x(:,2)).^2 + (y(:,3) - y(:,2)).^2 ); % length of the forearm

c3 = (a.^2 + b.^2 - L1.^2 - L3.^2)./(2.*L1.*L3);
%s3 = -(a.^2 + b.^2)./(a.*L3) - (a.^2).*b + b.^3 + b.*(L1.^2 + L3.^2)./(2.*a.*L1.*L3);
s3 = -(1/4 .* (L1.^4) - 1/2.*(L1.^2).*(L3.^2) - 1/2 .* (L1.^2) .* (b.^2) - 1/2.* (L1.^2).*(a.^2)...
+ 1/4 .* (L3.^4) - 1/2 .* (L3.^2) .* (b.^2) - 1/2 .* (L3.^2) .* (a.^2) + 1/4 .* (b.^4) + 1/2 .* (b.^2) .* (a.^2) + 1/4 .* (a.^4))...    
./( (L1.^2) .* (L3.^2));
c1_pos = (L3.*a)./(b.^2 + a.^2) - (-(L1.^2).*b + (L3.^2).*b - b.^3 - b.*(a.^2) )./(2.*L1.*(b.^2 + a.^2));
s1_pos = -(L3.*b)./(b.^2 + a.^2) - (-(L1.^2).*a + (L3.^2).*a - a.^3 - a.*(b.^2) )./(2.*L1.*(b.^2 + a.^2));

c1_neg = (L3.*a)./(b.^2 + a.^2) + (-(L1.^2).*b + (L3.^2).*b - b.^3 - b.*(a.^2) )./(2.*L1.*(b.^2 + a.^2));
s1_neg = -(L3.*b)./(b.^2 + a.^2) + (-(L1.^2).*a + (L3.^2).*a - a.^3 - a.*(b.^2) )./(2.*L1.*(b.^2 + a.^2));

%data_angles = [c3 s3 c1_pos s1_pos c1_neg s1_neg];
%data_angles = [smooth(atan2(s3,c3))];
%data_angles = [smooth(atan2(s1_pos,c1_pos))];
%data_angles = [smooth(atan2(s1_neg,c1_neg))];
data_angles = [atan2(s3,c3) atan2(s1_pos,c1_pos)];
data_pos = [abs(a) abs(b) L1 L3]; % normalized pose parameters with respect to shoulder position

% normalize the data angles and data pos between 0 and 1
%data_angles = (data_angles + pi)/(2*pi);
%data_pos = data_pos./norm(data_pos);

%data_angles = [smooth(atan2(s3,c3)) smooth(atan2(s1_neg,c1_neg))];
%data_angles = [smooth(atan2(s3,c3)) smooth(atan2(s1_pos,c1_pos)) smooth(atan2(s1_neg,c1_neg))];

%data_angle_traj = ComputeShapeOfTrajectory(data_angles,15);
end