function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


#注意在本类中样本点个数和中心点个数是不同的，不能混为一谈
min=0.0;
min_position=0.0;
i=0;
j=0;
for i=1:size(X,1)
  min=norm ((X(i,:)-centroids(1,:)),2)       #求两点之差的第二范数，也就相当于求两点的距离
  min_position = 1
  for j = 1:K
    len = norm ((X(i,:)-centroids(j,:)),2)
    if len<min
      min=len
      min_position=j
    endif
  endfor
  idx(i)=min_position
endfor






% =============================================================

end

