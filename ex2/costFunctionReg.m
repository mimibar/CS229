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
s=sigmoid(H);

% Cost & Gradient

j=1;
for i=1:m
    J=J -y(i)*log(s(i))-(1-y(i))*log(1-s(i));
    grad(j)= grad(j)+ (s(i)-y(i))*X(i,j);
end
J=J/m;
grad(j)= grad(j)/m ;

% you should not be regularizing the theta(1) parameter
% (which corresponds to theta0) in the code
% for j=2:t%n
%     J=J+(lambda/2*m)*theta(j)^2;
% end
T = theta;
T(1)=0;
T=T'*T;
J=J+(lambda/2*m)*T;

for j=2:t
    for i=1:m
        grad(j)= grad(j)+ (s(i)-y(i))*X(i,j);
    end
    grad(j)= grad(j)/m + (lambda/m)*theta(j);
end

% =============================================================

end
