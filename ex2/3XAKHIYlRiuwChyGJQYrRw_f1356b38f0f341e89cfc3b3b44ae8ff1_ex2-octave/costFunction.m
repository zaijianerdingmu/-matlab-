function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;   %初始化代价为0
grad = zeros(size(theta));  %初始化一个零向量grad，用来存储theta，其大小和theta一样，是个n维列向量

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z=  X * theta ;     %z是个m*1的向量
hx = 1./ ( 1 + exp(-1*z) )   %hx也是个m*1的向量

J = -1*(1/m)* [y'*log(hx)+(1-y)'*log(1-hx)]

grad = (1/m) * [(hx-y)' * X]'








% =============================================================

end
