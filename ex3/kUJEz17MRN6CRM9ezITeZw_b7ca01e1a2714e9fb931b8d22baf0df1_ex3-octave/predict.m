function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
a1=X;

a2 = 1./ ( 1 + exp(-1* (a1 * Theta1') ) );
a2= [ones(m, 1) a2];

a3 = 1./ ( 1 + exp(-1* (a2 * Theta2') ) );

%一定要完整指明返回值格式，再提取出我们需要的返回值而不能进行省略，比如错误示例[，p]= max(a3,[],2)，这会返回错误结果
[p1,p2] = max(a3,[],2);
p=p2;






% =========================================================================


end
