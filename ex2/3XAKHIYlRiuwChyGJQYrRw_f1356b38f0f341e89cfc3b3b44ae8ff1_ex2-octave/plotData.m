function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find Indices of Positive and Negative Examples


%X应该是一个m*2的矩阵，用于存储每个样本的特征x1和x2，也就是画图时的x坐标和y坐标。
%y应该是一个m*1的矩阵，用于存储样本的结果集，也就是每个样本的类别，是正类还是负类
pos = find(y==1);%pos是一个m*1的向量，find函数返回结果集y中样本类别为1（正类）的位置，按列从1开始数
neg = find(y ==0);%neg是一个m*1的向量，find函数返回结果集y中样本类别为0（负类）的位置，按列从1开始数
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
%X(pos, 1)表示点的横坐标，即X中那些pos指明的点的横坐标，即样本类别为1的横坐标
%X(pos, 2)表示点的纵坐标，即X中那些pos指明的点的纵坐标，即样本类别为1的纵坐标
%‘k+’表示画出的点为加号型，黑色
%'LineWidth', 2,表示线宽为2
%'MarkerSize', 7表示形状大小为7

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
%X(neg, 1)表示点的横坐标，即X中那些neg指明的点的横坐标，即样本类别为0的横坐标
%X(neg, 2)表示点的纵坐标，即X中那些neg指明的点的纵坐标，即样本类别为0的纵坐标
%‘ko’表示画出的点为圆点型
%'MarkerFaceColor', 'y'表示画出的点为黄色
%'MarkerSize', 7表示形状大小为7



% =========================================================================



hold off;

end
