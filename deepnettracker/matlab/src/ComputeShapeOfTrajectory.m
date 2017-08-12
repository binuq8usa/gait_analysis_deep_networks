function P_desc = ComputeShapeOfTrajectory(data, win_size)
% This computes local shape trajectory of the data using the method given
% in Dense Trajectories and Motion Boundaries (IJCV 2013/ CVPR 2011)
% data = [X Y] where number of rows gives observations

%finding the number of subsets
overlap = win_size - 1;
shift = win_size - overlap;
num_of_frames = size(data,1);
num_of_var = size(data,2);

sum_all = win_size;
num_of_subsets = 1;
while(sum_all <= num_of_frames)
    sum_all = win_size + shift*num_of_subsets;
    num_of_subsets = num_of_subsets+1;
end
num_of_subsets = num_of_subsets - 1;
P_desc = zeros(num_of_subsets, num_of_var * (win_size - 1));

for k = 1:1:num_of_subsets
    B = data(shift*(k-1) + 1 : shift*(k-1) + win_size,:);
    Tr = B(2:win_size,:) - B(1:win_size-1,:);
    Tr = Tr';
    %P_desc(k,:) = (Tr(:) ./ sum(abs(Tr(:))))';
    P_desc(k,:) = Tr(:)'; % not normalizing it 
    
    % Why not normalizing it?
    % normalization will place every trajectory 
    % Why it wont work well with normalization?
end

end

