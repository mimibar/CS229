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
h=X*theta ;%computing the matrix product Xtheta,

T = theta;
T(1)=0;%you should not regularize the T0 term.

%Cost (J)
J=h-y;
J=J'*J;
J=J/(2*m);
J=J+(lambda/(2*m))*(T'*T);



%gradient
B = h-y;
grad = X'*B;
grad=grad/m;
grad =grad +(lambda/m)*T;
grad = grad(:);







% =========================================================================

grad = grad(:);

end
