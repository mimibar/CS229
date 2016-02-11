function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));%the gradient of the cost

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
t=length(theta);
% H (a vector of all the hypothesis values for the entire training set) as 
% X * theta, with dimensions of (m x 1).
H=X*theta ;
s=sigmoid(H);

% Cost
for i=1:m
    J=J -y(i)*log(s(i))-(1-y(i))*log(1-s(i));
end
J=J/m;

%Gradient
for j=1:t
    for i=1:m
        grad(j)= grad(j)+ (s(i)-y(i))*X(i,j);
    end
    grad(j)= grad(j)/m;
end

% =============================================================

end

%testcase
%! X = [ones(4,1) magic(4)];
%! y = [1 0 1 0]';
%! theta=[-1 2 -3 4 -5]'
%!assert(costFunction(theta, X, y),0.693);
% j =  22.000
% g =
%   -0.25000
%   -5.25000
%    1.25000
%    1.50000
%   -6.00000
