function [xPCAWhite, k, U, S, V, XMean,pcomp] = pcaWhiten( X, portion,k )
% This function is analagous to the zca function except that we are only
% taking the most significant eigen vectors
if nargin < 2
    portion = 1;
end

if nargin < 3
    portion = 1;
    k = -inf;
end

rndidx = randperm(size(X, 1));
X_orig = X;
X = X(rndidx(1:round(size(X,1) * portion)), :);

m = mean(X, 1);
XMean = m;
Xc = bsxfun(@minus, X, m);

sigma = Xc' * Xc / size(Xc, 1);

[U, S, V] = svd(sigma, 0);

XRot = U' * Xc'; 

% selecting the number of eigen vectors automatically
if(k == -inf)
    lambda = sum(S,2);
    Sum = sum(lambda);
    temp = Sum;
    
    for ii = size(lambda,1):-1:1
        temp = temp - lambda(ii);
        if(temp / Sum < 0.99)
            k = ii;
            break;
        end
    end
end

xPCAWhite = diag(1./sqrt(diag(S(1:k,1:k)) + 10^-6)) * U(:,1:k)' * Xc';
xPCAWhite = xPCAWhite';

pcomp = U(:,1:k)' * Xc';
pcomp = pcomp';

end

