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
for i=1:m
    x=X(i,:)';%Lower-case x typically indicates a single training example.
    h= 1/(1+1/exp(- theta' * x));%Theta transpose h, a scalar for one training example
    
    %Cost
    J=J -y(i)*log(h)-(1-y(i))*log(1-h);
    
    %Gradient
    for j=1:t
        grad(j)= grad(j)+ (h-y(i))*X(i,j);
    end

end

J=J/m;
grad(j)= grad(j)/m;



% =============================================================

end
