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

t=length(theta);
% H (a vector of all the hypothesis values for the entire training set) as 
% X * theta, with dimensions of (m x 1).
H=X*theta ;
h=sigmoid(H);

% Cost & Gradient

j=1;
for i=1:m
    grad(j)= grad(j)+ (h(i)-y(i))*X(i,j);
end
%Since we want the sum of the products, we can use a vector multiplication.
%The size of each argument is (m x 1), and we want the vector product to be
%a scalar, so use a transposition so that (1 x m) times (m x 1) gives a
%result of (1 x 1), a scalar.
J=-y'*log(h)-(1-y)'*log(1-h)

J=J/m;%This is the unregularized cost.
grad(j)= grad(j)/m ;

% you should not be regularizing the theta(1) parameter
% (which corresponds to theta0) in the code
T = theta;
T(1)=0;
T=T'*T;
%Be sure you use enough sets of parenthesis to get the correct result.
J=J+(lambda/(2*m))*T;
for j=2:t
    for i=1:m
        grad(j)= grad(j)+ (h(i)-y(i))*X(i,j);
    end
    grad(j)= grad(j)/m + (lambda/m)*theta(j);
end

% =============================================================

end
