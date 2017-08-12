function [f,df] = congugateGradientRectifierNN(VV,Dim,XX,YY)
% This is same as the CG_CLASSIFY code provided by Hinton but has more flexibility
% in the number of layers 

% get the number of images in the batch
num_images = size(XX,1);
num_pair_layers = length(Dim)-1; % number of pair of layers

% get back the original weights
offset = 0;
for ll = 1:1:num_pair_layers
    w = reshape(VV(offset + 1: offset + (Dim(ll)+1)*Dim(ll+1)),Dim(ll)+1,Dim(ll+1));
    offset = offset + (Dim(ll)+1)*Dim(ll+1);
    W{ll} = w'; % same format as Hinton's code
end

XX_next = XX;
w_probs = cell(num_pair_layers,1); % with respect to input to another layer from a previous hidden layer with appending of bias units
data_inputs_per_layer = cell(num_pair_layers,1);

% sigmoid activation function % for RBM pair of layers
for u = 1:1:num_pair_layers-1
    data = [XX_next ones(num_images,1)];
    data_inputs_per_layer{u} = data;
    if(u ~= 1)
        w_probs{u-1} = data;
    end
    w = W{u}'; % getting the layer u weights
    
    % obtaining the output of layer u
    %data_a = 1./(1 + exp(-1*(data*w)));
    x = data*w;
    data_a = max(x,0); % rectilinear units
    
    % serving as input to next layer
    XX_next = data_a;
end

% output layer which is linear ( for function Approximator)
data = [XX_next ones(num_images,1)];
data_inputs_per_layer{num_pair_layers} = data;
if(num_pair_layers ~= 1)
    w_probs{num_pair_layers-1} = data;
end
w = W{num_pair_layers}';
data_a = data*w;
%data_a = 1./(1 + exp(-1*(data*w)));

YYout = data_a;

% compute the objective function for stacked auto-encoder
%f = -1/num_images*sum(sum( XX.*log(XXout) + (1-XX).*log(1-XXout)));
f = -1/num_images * sum(sum( (YY - YYout).^2)); % MSE Cost function for function approximator
%f = -1/num_images*sum(sum( YY.*log(YYout) + (1-YY).*log(1-YYout)));
%f = -sum(sum( YY.*log(YYout)));

% compute the derivates of the weights of the last layer
% df is same as delta_error / delta_weights = (delta_error /delta_outputnode) * (delta_outputnode /  delta_netoutputnode) * (delta_netoutputnode / delta_weights)
% For linear output layer : it becomes IO * 1 * data; 
% since netoutputnode = data * weights
IO = 1/num_images*(YYout-YY); % delta_E / delta_outputnode = -(target_outputnode - out_outputnode)
df = [];
Ix_last = IO;
dw = data'*Ix_last; % this is correct for output linear layer; delta_Outputnode / delta_NetOutputNode = outputnode * (1-outputnode) if sigmoidal function.
df = dw(:)';

% getting the original data
data = [XX ones(num_images,1)];

% Propagating the error backwards through the RBM layers
for u = num_pair_layers-1:-1:1
    w = W{u+1}'; % weights of the layers between l and l+1
    %Ix_last = (Ix_last*w').*w_probs{u}.*(1-w_probs{u}); % computing the error in this pair of layers or RBM 'u' % computing => (delta_error /delta_outputnode) * (delta_outputnode /  delta_netoutputnode)
    Ix_last = (Ix_last * w'); %(delta_error /delta_outputnode)
    delta_o_delta_net = (data_inputs_per_layer{u+1} > 0); % rectilinear
    delta_o_delta_net = double(delta_o_delta_net);
    Ix_last = Ix_last .* delta_o_delta_net;
    
    Ix_last = Ix_last(:,1:end-1);
    if(u ~= 1)
        dw = w_probs{u-1}'*Ix_last; % computing the update of the weights = > Ix_last * (delta_netoutputnode / delta_weights)
    else
        dw = data'*Ix_last;
    end
    df = [dw(:)' df];
end

df = df';


