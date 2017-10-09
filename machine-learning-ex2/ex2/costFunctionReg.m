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

z = X * theta;
h = sigmoid(z);
costs =  (-y .* log(h)) - ((1 - y) .* log(1 - h));
theta_without_t0 = theta;
theta_without_t0(1,1) = 0;
reg_cost = (lambda / (2 * m)) * sum(theta_without_t0.^2);
J = (sum(costs) / m) + reg_cost; 

err = (h - y);
n = length(theta);
err_n = repmat(err, 1, n);
grads = err_n .* X;
reg_grads = (lambda / m) * theta;
reg_grads(1,1) = 0;
grad = (sum(grads) / m) + reg_grads';


% =============================================================

end
