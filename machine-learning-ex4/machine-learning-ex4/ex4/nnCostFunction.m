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

% Add ones to the X data matrix
a1 = [ones(m, 1) X];

% compute output of the hidden layer
h_hidden = sigmoid(a1 * Theta1');

% Add ones to the output of hidden layer
h_hidden = [ones(m, 1) h_hidden];

% compute output of last layer
h = sigmoid(h_hidden * Theta2');

% convert y into boolean vectors of length K
encoder = 1:num_labels;
y_bool = (y == encoder);

% compute cost
costs =  (-y_bool .* log(h)) - ((1 - y_bool) .* log(1 - h));
J = sum(sum(costs)) / m;


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

DELTA_2 = zeros(size(Theta2));
DELTA_1 = zeros(size(Theta1));

% for each example
for t = 1:m
    
    % forward propagation 
    a1 = [1 X(t,:) ]'; % add 1 for the bias
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; % add 1 for the bias
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % error of output layer
    y3 = y_bool(t,:)';
    delta_3 = a3 - y3;
    
    % error of hidden layer
    delta_2 = Theta2' * delta_3;
    delta_2 = delta_2(2:end); % remove unused error on bias
    delta_2 = delta_2 .* sigmoidGradient(z2);
    
    % accumulate gradient
    DELTA_2 = DELTA_2 + delta_3 * a2';
    DELTA_1 = DELTA_1 + delta_2 * a1';
    
end

Theta2_grad = DELTA_2 / m;
Theta1_grad = DELTA_1 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% regularization term
theta1_without_bias = Theta1(:, 2:end);
theta2_without_bias = Theta2(:, 2:end);

reg = 0;
reg = reg + sum(sum((theta1_without_bias .^ 2)));
reg = reg + sum(sum((theta2_without_bias .^ 2)));
reg = reg * (lambda / (2 * m));

J = J + reg;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
