function [xPCAWhite,pcomp] = pcaWhitenWithoutSVD( X,k,U,S,X_mean )
% This function is analagous to the zca function except that we are only
% taking the most significant eigen vectors

Xc = bsxfun(@minus, X, X_mean);

xPCAWhite = diag(1./sqrt(diag(S(1:k,1:k)) + 10^-6)) * U(:,1:k)' * Xc';
xPCAWhite = xPCAWhite';

pcomp = U(:,1:k)' * Xc';
pcomp = pcomp';

end

