function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %num_iters参数的意思是迭代次数
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);      %_history是一个列向量，其行数为迭代次数

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  theta = theta - alpha* (1/m)* (( X*theta - y)'* X)';






    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);    %将J_history的第iter行设置为我们更新theta值

end

end
