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

z =  X * theta;
h = sigmoid(z);
oneM =ones(size(h));

j = - y .* log(h) - (oneM - y) .* log(oneM-h);
J = sum(j)/m;
J = J + sum( theta(2:size(theta)).* theta(2:size(theta)) )*lambda/(2*m);

grad(1) = sum((h - y) .* X(:,1))/m;
grad(2) = (sum((h - y) .* X(:,2)) + lambda * theta(2))/m;
grad(3) = (sum((h - y) .* X(:,3)) + lambda * theta(3))/m;

% =============================================================

end
