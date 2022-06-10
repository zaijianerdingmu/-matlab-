function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

%theta矩阵的行数是num_labels个表示种类个数，theta矩阵的列数是n+1个，因为有n个特征，还有一列纯1
m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the
%       max element, for more information see 'help max'. If your examples
%       are in rows, then, you can use max(A, [], 2) to obtain the max
%       for each row.
%

p1 = 1./ ( 1 + exp(-1* (X * all_theta') ) );

%matlab函数返回值很死板，只有我们指定额外返回值时，matlab才会返回除了默认的还有我们想要的返回值
%当指定返回值为[p1,p2]这样的返回值形式后，max函数返回两列向量，第一列是最大值，第二列是最大值对应的索引
%至于返回的是每行的最大值还是每列的最大值，由max函数的第三个参数来决定，1表示返回每列最大值，2表示返回每行最大值
[p1,p2]=max(p1,[],2);

p=p2;



% =========================================================================


end
