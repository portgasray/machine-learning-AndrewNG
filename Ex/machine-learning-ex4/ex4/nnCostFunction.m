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

% input_layer_size 400
% hidden_layer_size here is 25
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% why tetha1 belongs to (25, 401)??? £¨S(j+1), Sj + 1£©

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% theta2 (10,26)

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
X_1 = [ones(m,1), X];          % (5000, 401)
a2 = sigmoid(X_1 * Theta1');   % (5000, 25)
a2 = [ones(m,1), a2];        % (5000, 26)
a3 = sigmoid(a2 * Theta2');   % (5000, 10)

J_without_regular = 0;

%initialize a martix of y size :(5000, 10) each row is a labeled with y_k = 1
y_labeled = zeros(m, num_labels);

for c = 1 : m
  y_labeled(c, y(c)) = 1;
  J(c) = (1/m) * ( -y_labeled(c,:) * log( a3(c,:)' ) - ...
                  ( 1 - y_labeled(c,:) ) * log( 1 - a3(c,:)' ) );
  J_without_regular += J(c);
endfor

% sum(x(:))  equals sum(sum(x , 1)) and sum(sum(x , 2))
J = J_without_regular + (lambda/(2*m)) *(sum(Theta1(:,2:end)(:).^2) + ...
                                          sum(Theta2(:,2:end)(:).^2) );

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
delta_2 = zeros(1 + hidden_layer_size, m);
delta_3 = zeros(num_labels, m);

Delta_1 = zeros(hidden_layer_size, 1 + input_layer_size);
Delta_2 = zeros(num_labels, 1 + hidden_layer_size);
for t = 1:m
  
  % step 1: forward pass: camputing the activations
  a_1 = [1,X(t,:)] ;    % 1, 401
  z_2 = Theta1 * a_1';  % (25, 401) * (401, 1)
  a_2 = sigmoid(z_2);   % 25, 1
  a_2 = [1; a_2];       % 26, 1
  z_3 =  Theta2 * a_2;   % (10, 26) * (26, 1)
  a_3 = sigmoid(z_3);   % 10, 1
   
  %%Backpropagation: compute an "error term" delta_layer
  %step 2: layer 3(the output layer) compute error
  for k = 1 : num_labels
    delta_3(k,t) = a_3(k,:) - (y(t) == k);  %layer = 3, where y belongs to {0,1}
  endfor
  
  %step 3: hidden layer (l = 2) (25, 10) * (10,1) .* (26,1) 
  %(26, 1)
  %delta_2(:, t) = Theta2' * delta_3(:, t) .* sigmoidGradient(a_2);
  delta_2(:, t) = Theta2' * delta_3(:, t) .* (a_2).*(1-a_2);       %% do not need sigmoid again
  %% no delta_1
  
  %step 4: accumulate the gradient
  Delta_1 = Delta_1 + delta_2(2:end, t) * a_1; %% remove delta_2(0), the bias unit
  Delta_2 = Delta_2 + delta_3(:, t) * a_2';
  
  %delta = [delta_2(2:end); delta_3];  %hidden_layer_size + num_labels
  %a = [a_1(2:end)'; a_2(2:end)];      %input_layer_size + hidden_layer_size
  %Delta =  Delta + delta * a';        %% hidden_layer_size + num_labels, input_layer_size + hidden_layer_size
  
endfor

%%step 5: divide the accumulated gradients by m to obtain the gradients for the NN costFunction
%Theta1_grad = (1/m)* Delta_1;
%Theta2_grad = (1/m)* Delta_2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,1) = (1/m) * Delta_1(:,1);
Theta1_grad(:,2:end) = (1/m) * (Delta_1(:,2:end) + lambda * Theta1(:,2:end) );
Theta2_grad(:,1) = (1/m) * Delta_2(:,1);
Theta2_grad(:,2:end) = (1/m) * (Delta_2(:,2:end) + lambda * Theta2(:,2:end) );
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
