function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% if z is matrix
% z =  X * (theta) : (100,3) * (3,1) = (100,1);
g =  1./(1 + e.^(-z));

% =============================================================

end
