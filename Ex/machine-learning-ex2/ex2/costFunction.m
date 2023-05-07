function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% we have h_theta belongs (100,1), y  belongs(100,1), and calculate y(1) * h_theta(x(1))
% finally we have matrix (100,100) that means h_theta * y'
% hypothesis function equals g( theta' * x)
% 1st, calculate z matrix, equals  - (X * theta) belongs (100,3) * (3,1) = (100,1)
z = X * theta ;

% 2nd, get (100,1) matrix that means y(i)' * log(h_theta(x(i))), 
% sum function is unnecessary, matrix could add automatically

J=(1/m)*( (-y)' * log(sigmoid(z)) - ((1-y)' * log(1-sigmoid(z))) );

% gradient descent: X (m_sample, n_feature) * hypothesis function
% (1, m) * (m, 1)
%grad(0) = (1/m) * ( X(:,1)' * (sigmoid(z) - y) );
%grad(1) = (1/m) * ( X(:,2)' * (sigmoid(z) - y) );
%grad(2) = (1/m) * ( X(:,3)' * (sigmoid(z) - y) );

grad = (1/m) * ( X' * (sigmoid(z) - y) );
% =============================================================

end
