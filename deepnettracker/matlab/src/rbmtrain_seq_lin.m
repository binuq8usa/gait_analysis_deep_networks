% this function is to train an rbm with a set of features computed for a
% set of sequences.
% each batch here would then correspond to the set of features from a
% single sequence
% the set of sequences are combined and can be indexed using the
% opts.batch_size
function rbm = rbmtrain_seq_lin(rbm,X,opts,flag_seq)

rand('state',0);
% set of commands taken from the original rbmtrain in toolbox
assert(isfloat(X), 'x must be a float');
assert(all(X(:)>=0) && all(X(:)<=1), 'all data in x must be in [0:1]');

if(flag_seq) % training for a sequence
    % getting the number of frames per sequence
    Size = cumsum(opts.seq_size);
    numbatches = length(opts.seq_size);
else
    num_images = size(X,1);
    numbatches = num_images / opts.batch_size;
end

% set the gpu mode arrays for data and rbm
X_gpu = single(gpuArray(X));
W_gpu = single(gpuArray(rbm.W));
bh_gpu = single(gpuArray(rbm.bh));
bv_gpu = single(gpuArray(rbm.bv));
del_W_gpu = single(gpuArray(rbm.del_W));
del_bh_gpu = single(gpuArray(rbm.del_bh));
del_bv_gpu = single(gpuArray(rbm.del_bv));
momentum_gpu = single(gpuArray(rbm.momentum));
epsilon_w_gpu = single(gpuArray(rbm.epsilon_w));
epsilon_vc_gpu = single(gpuArray(rbm.epsilon_vc));
epsilon_vb_gpu = single(gpuArray(rbm.epsilon_vb));
weightcost_gpu = single(gpuArray(rbm.weightcost));
    
% for each epoch
err_graph = zeros(opts.numepochs,1);

% for each epoch
for k = 1:1:opts.numepochs

    if(flag_seq)
        % create a different ordering in which the sequence 
        seq_num = randperm(numbatches);
    else
        seq_num = randperm(num_images);
    end
    
    err_sum = 0;
    for l = 1:1:numbatches
        if(flag_seq)
            seq_idx = seq_num(l);
            if(seq_idx == 1)
                start_idx = 1;
                end_idx = Size(seq_idx);
            else
                start_idx = Size(seq_idx-1) + 1;
                end_idx = Size(seq_idx);
            end
            % check for empty sequences
            if(start_idx-1 == end_idx)
                continue;
            end
            
            % getting the features for a sequence
            x_batch = X_gpu(start_idx:end_idx,:);
        else
            x_batch = X_gpu(seq_num((l - 1) * opts.batch_size + 1 : l * opts.batch_size), :);
        end

        %%%%% POSITIVE PHASE
        v0 = x_batch;
        num_images_per_batch = size(x_batch,1);
        num_hid = size(rbm.bh,1);
        
        % compute the conditional probability that P(H_j = 1| v) for the
        % per-frame features of each sequence
        prob_Hj_V_pos = v0 * W_gpu' + repmat(bh_gpu',num_images_per_batch,1);
        
        % <V_i*H_j> sampled from P(H|V)P(V)
        Vi_Hj_pos = (v0' * prob_Hj_V_pos)'; 
        
        % H_j sampled from P(H|V)P(V)
        Hj_pos = (sum(prob_Hj_V_pos))'; 
        
        % V_i sampled from P(V)
        Vi_pos = (sum(v0))';
        
        % computing the hidden states
        pos_Hj_states = prob_Hj_V_pos + single(gpuarray(randn(num_images_per_batch,num_hid)));
        
        % equivalent command 
        %h0 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
        
        %%%%% NEGATIVE PHASE
        v1 = 1./(1 + exp( -1 * (single(pos_Hj_states) * W_gpu + repmat(bv_gpu',num_images_per_batch,1))));
        % compute P(H_j = 1|v) for negative phase
        prob_Hj_V_neg = v1 * W_gpu' + repmat(bh_gpu',num_images_per_batch,1);
        
        % <V_i*H_j> for negative phase
        Vi_Hj_neg = (v1' * prob_Hj_V_neg)';
        
        % H_j sampled from P(V,H)
        Hj_neg = (sum(prob_Hj_V_neg))';
        
        % V_j sampled from negative phase
        Vi_neg = (sum(v1))';
        
        % compute the error between the original data and the reconstructed
        % data
        err_1 = sum( sum((v0 - v1).^2))/num_images_per_batch; 
        err_sum = err_sum + err_1;   
        
        %%%% UDPATE RULE for weights and biases
        del_W_gpu = momentum_gpu * del_W_gpu + epsilon_w_gpu * ( (Vi_Hj_pos - Vi_Hj_neg)/num_images_per_batch - weightcost_gpu* W_gpu );
        del_bh_gpu = momentum_gpu * del_bh_gpu + (epsilon_vc_gpu/num_images_per_batch) * (Hj_pos - Hj_neg);
        del_bv_gpu = momentum_gpu * del_bv_gpu + (epsilon_vb_gpu/num_images_per_batch) * (Vi_pos - Vi_neg);

        W_gpu = W_gpu + del_W_gpu;
        bh_gpu = bh_gpu + del_bh_gpu;
        bv_gpu = bv_gpu + del_bv_gpu;
                
        %fprintf('Epoch %d - Batch - %d ; Error = %f\n',k,l,err_1);
    end
    
    err_graph(k) = err_sum;
    fprintf('Error at iteration %d = %f\n',k,err_sum/numbatches);
    
end

% get all the rbm values back
X = gather(X_gpu);
rbm.W = gather(W_gpu);
rbm.bh = gather(bh_gpu);
rbm.bv = gather(bv_gpu);
rbm.del_W = gather(del_W_gpu);
rbm.del_bh = gather(del_bh_gpu);
rbm.del_bv = gather(del_bv_gpu);
rbm.momentum = gather(momentum_gpu);
rbm.epsilon_w = gather(epsilon_w_gpu);
rbm.epsilon_vc = gather(epsilon_vc_gpu);
rbm.epsilon_vb = gather(epsilon_vb_gpu);
rbm.weightcost = gather(weightcost_gpu);




end

