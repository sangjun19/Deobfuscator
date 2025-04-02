// Repository: MeetNiceMe/matlab
// File: 基于 K-means 聚类算法的图像区域分割/searchintial.m

function C=searchintial(X,method,varargin)
switch lower(method(1))
    case 's' 
        K=varargin{1};
        C=X(randsample(size(X,1),K),:);
    case 'u' 
        Xmins=min(X,[],1);
        Xmaxs=max(X,[],1);
        K=varargin{1};
        C=unifrnd(Xmins(ones(K,1),:), Xmaxs(ones(K,1),:));
end