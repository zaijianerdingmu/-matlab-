function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=  X * theta ;     %z是个m*1的向量
hx = 1./ ( 1 + exp(-1*z) )   %hx也是个m*1的向量
J = -1*(1/m)* [y'*log(hx)+(1-y)'*log(1-hx)] + lambda *(1/(2*m))* [theta(2:size(theta),:)' * theta(2:size(theta),:)]

grad1 = (1/m) * [(hx-y)' * X]'
grad2 = (1/m) * [(hx-y)' * X]' + (lambda / m) *theta;

grad(1,:) = grad1(1,:);
grad(2:size(theta),:) = grad2(2:size(theta),:);



% =============================================================

end
