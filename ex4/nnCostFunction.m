function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = X;                 %将输入单元直接设置为a1，因为输入单元是直接的输入并不需要激活函数
a1 = [ones(m, 1) X];    %为输入单元加上一列全1，也就是偏置单元

a2 = 1./ ( 1 + exp(-1* (a1 * Theta1') ) );      %输入单元经过一个矩阵后在套上激活函数形成一层隐藏单元
a2= [ones(m, 1) a2];    %为当前层的隐藏单元加上一个偏执单元

a3 = 1./ ( 1 + exp(-1* (a2 * Theta2') ) );      %隐藏单元经过一个矩阵后再套上激活函数形成输出单元


%将函数传过来的标签向量改变一下形式，由单纯的一列，每个元素是类别序号转换成：
%行数是样本个数，列数是种类个数，每一行存储一个样本对应的类别，如果是第i类别，则就在当前行的第i列置1，其余列置0
Y = zeros(m, size(X,2) )
for i = 1 : m
  Y( i , y(i) ) = 1;
end



%在以前的只有一个输出单元的情况下，我们代价函数的公式是y并没有下标k
%而现在有了k个输出单元，代价函数的公式中y有了下标k并且对k进行累加
%那么我们有了一个思路：先单独看一个特征，只算一个输出单元。体现在矩阵上也就是先算X矩阵的一列和Y矩阵的一列，
%那么在单独算一个特征的时候我们的代价函数就与原先只有一个输出单元的情况相同
%我们再算每个单独特征的代码外面加一个for循环，算出每个特征对应的代价并累加到J上，那么我们的J也就是整个样本所有特征计算出的代价
for i = 1 : size(a3,2)    %遍历每个特征
  a3_temp = a3( : , i)    %取hx中的一列
  Y_temp = Y( : , i)      %取标签中的一列
  J += -1*(1/m)* [Y_temp'*log(a3_temp)+(1-Y_temp)'*log(1-a3_temp)]   %对指定列即指定特征进行代价函数计算并累加到J上
end



%注意Theta矩阵的行数是下一层单元的个数，Theta矩阵的列数是当前层矩阵单元的个数+1，
%因为偏执单元也需要一个系数，
%Theta的行下标是从1开始，因为我们从1开始对应下一层的第一个单元，而不包含下一层的偏执单元
%Theta的行下标从1开始，到下一层单元数个数结束。总计的行数正好是下一层单元数
%Theta的列下标是从0开始，因为我们从0开始为了匹配当前层的偏执单元X0
%Theta的列下标从0开始。到当前层单元的个数结束。总计的列数正好是当前层单元个数+1

%注意我们正则化时是不包含对偏执单元上的系数的，而每个Theta矩阵的第一列都是当前层的偏执单元的
%系数，因此我们在进行系数平方累加时，跳出矩阵的第一列即可。


%用循环算出Theta1系数矩阵的正则化项
costTheta1=0;
for i = 1 : size(Theta1,1)
  for j = 2 : size(Theta1,2)
    costTheta1 += ( Theta1(i,j) *Theta1(i,j) )
  endfor
end

%用循环算出Theta2系数矩阵的正则化项
costTheta2=0;
for i = 1 : size(Theta2,1)
  for j = 2 : size(Theta2,2)
    costTheta2 += ( Theta2(i,j) * Theta2(i,j) )
  endfor
end

J = J + lambda*(1/(2*m))*(costTheta1 + costTheta2);

for t = 1:m,                        %遍历每一个样本，将每个样本正向传播算出每个节点的激活值再反向传播算出每个节点的误差值，通过激活值和误差值组合累加来算出关于每个参数的梯度
  z1 = [1,X(t,:)]';                 %取出样本集中的单个样本，加上偏置值1后再进行转置形成一个关于单个样本的列向量，其列数为特征数加1
  a1=z1;                            %对于输入层而言，特征值就是激活值

  z2=Theta1 * a1;                   %为第一层的激活值乘上参数作为下一层的输入
  a2 = 1./ ( 1 + exp(-1* z2 ) );    %套上激活函数后形成第二层即隐藏层的激活值
  a2= [1; a2];                      %为当前层的隐藏单元加上一个偏执单元

  z3=Theta2 *a2;                    %为当前层的激活值乘上参数作为下一层的输入
  a3 = 1./ ( 1 + exp(-1* z3 ) );    %套上激活函数后形成第三层即输出层的激活值



  delta3 = a3 - Y(t,:)'             %a3就是第t个样本正向传播算出的hx，我们从Y中找出第t个样本的真实标签，并转置成列向量，由反向传播算法可知，输出层的误差delta就是激活值减去标签值

  delta2 = Theta2' * delta3 .* a2 .* ( 1 - a2 )    %再根据反向传播算法公式从后往前依次算每层单元的误差delta，但是第一层我们不用算，因为第一层是我们的直接输入，没有误差

  Theta2_grad += delta3*a2'       %对应单个样本而言，当前层参数的梯度就是下一层的误差值乘上当前层的激活值的转置，要是所有样本那就进行累加和

  Theta1_grad += delta2(2:end,:)*a1'    %注意题目要求我们并不需要计算关于第一层第一个单元即偏执单元参数的梯度，因此我们就跳过该单元的梯度计算，即运算时跳过误差矩阵的第一行
end

  Theta1_grad = Theta1_grad * (1/m);    %注意最后的梯度需要累加和再除以样本数量m
  Theta2_grad = Theta2_grad * (1/m);



%算出梯度后我们再对梯度进行正则化
%与上述代价函数的正则化相同，每个Theta_grad矩阵的第一列都是关于偏置单元的梯度值，因此我们正则化时跳出矩阵第一列即可
for i = 1 : size(Theta1_grad,1)
  for j = 2 : size(Theta1_grad,2)
    Theta1_grad(i,j) += ( (lambda/m)*Theta1(i,j) )
  endfor
end

for i = 1 : size(Theta2_grad,1)
  for j = 2 : size(Theta2_grad,2)
    Theta2_grad(i,j) += ( (lambda/m)*Theta2(i,j) )
  endfor
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
