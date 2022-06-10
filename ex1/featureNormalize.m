function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;  %用于存储特征缩放后的X并返回
mu = zeros(1, size(X, 2));       %一个一行（特征数）列的向量，用于存储每个特征的平均值
sigma = zeros(1, size(X, 2));    %一个一行（特征数）列的向量,

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

mu = mean(X); %计算X矩阵中各列的平均值并储存在一行，（特征数）列的行向量中
sigma = std(X); %计算X矩阵中各列的标准差并存储在一行，（特征数）列的向量中

for i = 1:size(X,2),       %设置循环数为特征数，用i来遍历每列
  for j = 1:size(X,1),     %设置循环数为样本个数，用j来遍历每行
    X_norm(j,i) = X_norm(j,i) - mu(i);   %将每行的指定i特征减去该列的平均值mu(i)
  end;
end;


for i = 1:size(X,2),        %设置循环数为特征数，用i来遍历每列
  for j = 1:size(X,1),      %设置循环数为样本个数，用j来遍历每行
    X_norm(j,i) = X_norm(j,i) / sigma(i);     %将每行的指定i特征减去该列的标准差sigma(i)
  end;
end;






% ============================================================

end
