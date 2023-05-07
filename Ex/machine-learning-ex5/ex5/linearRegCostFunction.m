function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%non-vectorize
%J = (1/2*m)* sum((X * theta - y).^2, 1) + (lambda/2*m) * theta(2:end)'*theta(2:end) ;

%% vectorized
J = (1/(2*m)) * (X*theta - y)' * (X*theta - y) + ...
      (lambda/(2*m)) * theta(2:end)'*theta(2:end) ;

%% non-vectorized way implement Gradient descent of Linear Regression
%% X(:,2:end) 12, 1
grad(1) = (1/m) * X(:,1)' * (X * theta - y);
grad(2:end) = (1/m) * X(:,2:end)' * (X * theta - y)  + (lambda/m) * theta(2:end);

%% vectorized
%grad =  (1/m) * X' * (X * theta - y);







% =========================================================================

grad = grad(:);

end
