function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

iterations =  1500;

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %%way 1: non-vectorize way
    %temp0 = theta(1) - alpha * (1/m) * sum(X*theta - y);
    %temp1 = theta(2) - alpha * (1/m) * sum((X*theta - y) .* X(:,2));
    %theta = [temp0; temp1];
    
    %% way 2: vectorize way
    theta = theta - (alpha/m) * X' * (X * theta - y);
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    end

end
